import torch
import os
import sys
import numpy as np
import torch.distributed as dist
import logging
import time
import json
import re
from datetime import datetime, timedelta
import h5py
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from onescience.models.pangu import Pangu
from onescience.datapipes.climate import ERA5Datapipe
from onescience.utils.YParams import YParams
from onescience.memory.checkpoint import replace_function
from apex import optimizers


FORECAST_STEPS = 8
FORECAST_INTERVAL_HOURS = 6


def loss_func(x, y, weights, level_weight=1.0):
    return level_weight * (F.l1_loss(x, y, reduction='none') * weights).mean()


def get_stats(data_dir, channels):
    """读取与 ERA5Datapipe 相同的变量顺序、均值和标准差。"""
    with open(os.path.join(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    all_variables = metadata["variables"]
    channel_indices = [all_variables.index(name) for name in channels]
    stats_dir = os.path.join(data_dir, "stats")
    means = np.load(os.path.join(stats_dir, "global_means.npy"))
    stds = np.load(os.path.join(stats_dir, "global_stds.npy"))
    means = means[:, channel_indices].astype(np.float32)
    stds = stds[:, channel_indices].astype(np.float32)
    if np.any(stds == 0):
        raise ValueError("global_stds.npy 中存在标准差为 0 的变量")
    return channel_indices, means, stds


def _flatten_values(value):
    if isinstance(value, bytes):
        return [value.decode("utf-8")]
    if isinstance(value, str):
        return [value]
    if torch.is_tensor(value):
        return [str(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flatten_values(item))
        return result
    return [str(value)]


def extract_t0_names(filename_info, batch_size):
    """从 data[4] 中提取当前 batch 的 T0 名称。"""
    # ERA5Datapipe 的 data[4][0] 是输入样本名，data[4][-1] 通常是单步标签名。
    input_names = (
        filename_info[0]
        if isinstance(filename_info, (list, tuple))
        else filename_info
    )
    names = []
    for value in _flatten_values(input_names):
        match = re.search(r"(\d{10})", os.path.basename(value))
        if match:
            names.append(match.group(1))

    if len(names) != batch_size:
        raise ValueError(
            f"无法从 data[4] 提取 {batch_size} 个 T0，"
            f"提取结果={names}，原始 data[4]={filename_info}"
        )
    return names


def get_label_path(t0_name, step, data_dir, allowed_years):
    t0 = datetime.strptime(t0_name, "%Y%m%d%H")
    valid_time = t0 + timedelta(
        hours=(step + 1) * FORECAST_INTERVAL_HOURS
    )
    valid_name = valid_time.strftime("%Y%m%d%H")
    valid_year = valid_time.year
    if valid_year not in allowed_years:
        return None, valid_name
    path = os.path.join(
        data_dir, "data", str(valid_year), f"{valid_name}.h5"
    )
    return path, valid_name


def rollout_labels_available(t0_names, data_dir, allowed_years):
    """检查一个 batch 的 +6h～+48h 标签是否全部存在且属于当前数据划分。"""
    for t0_name in t0_names:
        for step in range(FORECAST_STEPS):
            path, _ = get_label_path(
                t0_name, step, data_dir, allowed_years
            )
            if path is None or not os.path.isfile(path):
                return False
    return True


def all_ranks_have_labels(available, local_rank):
    """DDP下确保所有进程同时执行或同时跳过一个 batch。"""
    if not dist.is_initialized():
        return available
    flag = torch.tensor(
        1 if available else 0,
        device=local_rank,
        dtype=torch.int32,
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def load_target_batch(t0_names, step, data_dir, allowed_years,
                      channel_indices, means, stds):
    """只读取当前一个预报步的真实标签，返回 [B,69,H,W] 归一化 Tensor。"""
    labels = []
    for t0_name in t0_names:
        path, valid_name = get_label_path(
            t0_name, step, data_dir, allowed_years
        )
        if path is None or not os.path.isfile(path):
            raise FileNotFoundError(
                f"T0={t0_name} 缺少 +{(step + 1) * 6}h 标签 "
                f"{valid_name}: {path}"
            )

        with h5py.File(path, "r") as f:
            label = f["fields"][:].squeeze()

        if label.ndim != 3:
            raise ValueError(
                f"{path} 的 fields 应为 [C,H,W]，实际为 {label.shape}"
            )
        if label.shape[0] == len(channel_indices):
            # 文件本身已经只包含目标69通道。
            label = label.astype(np.float32, copy=False)
        else:
            label = label[channel_indices].astype(np.float32, copy=False)

        label = (label - means[0]) / stds[0]
        labels.append(label.astype(np.float32, copy=False))

    return torch.from_numpy(np.stack(labels, axis=0))


def autoregressive_train_loss(model, initial_state, t0_names, surface_mask,
                              surface_weights, pressure_weights, data_dir,
                              allowed_years, channel_indices, means, stds,
                              local_rank, world_size):
    """8步自回归训练；每步预测作为下一步输入，并采用逐步截断反传。"""
    state = initial_state.to(local_rank, dtype=torch.float32)
    static = surface_mask[:state.shape[0]]
    loss_value = 0.0

    for step in range(FORECAST_STEPS):
        model_input = torch.cat(
            [state[:, :4], static, state[:, 4:]], dim=1
        )
        with replace_function(
            model, ["layer2", "layer3"], world_size > 1
        ):
            out_surface, out_upper_air = model(model_input)

        # 根据 T0 只读取当前时效的真实 HDF5，归一化后再放入 GPU。
        target = load_target_batch(
            t0_names=t0_names,
            step=step,
            data_dir=data_dir,
            allowed_years=allowed_years,
            channel_indices=channel_indices,
            means=means,
            stds=stds,
        ).to(
            local_rank, dtype=torch.float32, non_blocking=True
        )
        tar_surface = target[:, :4]
        tar_upper_air = target[:, 4:]
        out_upper_air = out_upper_air.reshape(tar_upper_air.shape)

        step_loss = (
            loss_func(
                out_surface, tar_surface, surface_weights,
                level_weight=0.25
            )
            + loss_func(
                out_upper_air, tar_upper_air, pressure_weights,
                level_weight=1.0
            )
        )

        # 对 8 个时效取平均；逐步 backward 可及时释放当前步计算图。
        (step_loss / FORECAST_STEPS).backward()
        loss_value += step_loss.detach().item() / FORECAST_STEPS

        # 模型预测（归一化空间）作为下一步输入。
        # detach 截断跨时效计算图，避免 8 步全分辨率反传导致显存爆炸。
        state = torch.cat([out_surface, out_upper_air], dim=1).detach()

        del model_input, target, tar_surface, tar_upper_air
        del out_surface, out_upper_air, step_loss

    return loss_value


@torch.no_grad()
def autoregressive_valid_loss(model, initial_state, t0_names, surface_mask,
                              surface_weights, pressure_weights, data_dir,
                              allowed_years, channel_indices, means, stds,
                              local_rank, world_size):
    """验证阶段连续滚动 8 步，返回 8 个时效的平均 loss。"""
    state = initial_state.to(local_rank, dtype=torch.float32)
    static = surface_mask[:state.shape[0]]
    loss_value = 0.0

    for step in range(FORECAST_STEPS):
        model_input = torch.cat(
            [state[:, :4], static, state[:, 4:]], dim=1
        )
        with replace_function(
            model, ["layer2", "layer3"], world_size > 1
        ):
            out_surface, out_upper_air = model(model_input)

        target = load_target_batch(
            t0_names=t0_names,
            step=step,
            data_dir=data_dir,
            allowed_years=allowed_years,
            channel_indices=channel_indices,
            means=means,
            stds=stds,
        ).to(
            local_rank, dtype=torch.float32, non_blocking=True
        )
        tar_surface = target[:, :4]
        tar_upper_air = target[:, 4:]
        out_upper_air = out_upper_air.reshape(tar_upper_air.shape)
        step_loss = (
            loss_func(
                out_surface, tar_surface, surface_weights,
                level_weight=0.25
            )
            + loss_func(
                out_upper_air, tar_upper_air, pressure_weights,
                level_weight=1.0
            )
        )
        loss_value += step_loss.item() / FORECAST_STEPS
        state = torch.cat([out_surface, out_upper_air], dim=1)

        del model_input, target, tar_surface, tar_upper_air
        del out_surface, out_upper_air, step_loss

    return loss_value


def main():

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()

    ## Model config init
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")

    ## Distributed config init
    cfg.world_size = 1
    if "WORLD_SIZE" in os.environ:
        cfg.world_size = int(os.environ["WORLD_SIZE"])
    world_rank = 0
    local_rank = 0
    if cfg.world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = dist.get_rank()

    cfg_data = YParams(config_file_path, "datapipe")
    datapipe = ERA5Datapipe(params=cfg_data, distributed=dist.is_initialized())
    train_dataloader, train_sampler = datapipe.train_dataloader()
    val_dataloader, val_sampler = datapipe.val_dataloader()

    # dataloader 只提供归一化后的 T0 输入和 T0 名称；8个真实标签根据
    # T0 名称从 HDF5 直接读取，并使用同一组统计量归一化。
    channel_indices, means, stds = get_stats(
        cfg_data.dataset.data_dir,
        cfg_data.dataset.channels,
    )
    train_years = set(int(year) for year in cfg_data.dataset.train_ratio)
    val_years = set(int(year) for year in cfg_data.dataset.val_ratio)

    surface_weights = torch.as_tensor(cfg_data.dataset.weights[:4], device=local_rank, dtype=torch.float32).view(1, -1, 1, 1)
    pressure_weights = torch.as_tensor(cfg_data.dataset.weights[4:], device=local_rank, dtype=torch.float32).view(1, -1, 1, 1)

    land_mask = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "land_mask.npy")).astype(np.float32))
    soil_type = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "soil_type.npy")).astype(np.float32))
    topography = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "topography.npy")).astype(np.float32))
    topography = (topography - topography.mean()) / (topography.std(unbiased=False) + 1e-6)
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0).to(local_rank)
    surface_mask = surface_mask.unsqueeze(0).repeat(cfg_data.dataloader.batch_size, 1, 1, 1)

    ## Model init
    model = Pangu(img_size=cfg_data.dataset.img_size,
                  patch_size=cfg.patch_size,
                  embed_dim=cfg.embed_dim,
                  num_heads=cfg.num_heads,
                  window_size=cfg.window_size,
                  ).to(local_rank)
    optimizer = optimizers.FusedAdam(model.parameters(), betas=(0.9, 0.999), lr=5e-4, weight_decay=3e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    ## Train process init
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    train_loss_file = f"{cfg.checkpoint_dir}/trloss.npy"
    valid_loss_file = f"{cfg.checkpoint_dir}/valoss.npy"
    best_valid_loss = 1.0e6
    best_loss_epoch = 0
    train_losses = np.empty((0,), dtype=np.float32)
    valid_losses = np.empty((0,), dtype=np.float32)

    ## Get model params count
    if cfg.world_size == 1:
        total_params = sum(p.numel() for p in model.parameters())
        print("\n\n")
        print("-" * 50)
        print(f"📂 now params is {total_params}, {total_params / 1e6:.2f}M, {total_params / 1e9:.2f}B")
        print("-" * 50, "\n")

    ## Load model weight if there exist well-trained model 
    if os.path.exists(f"{cfg.checkpoint_dir}/model_bak.pth"):
        if world_rank == 0:
            print("\n\n")
            print("-" * 50)
            print(f"✅ There has a model weight, load and continue training...")
            print(f'If you want to train a new model, ensure there is no *.pth file in {cfg.checkpoint_dir}')
            print("-" * 50, "\n")
        ckpt = torch.load(f"{cfg.checkpoint_dir}/model_bak.pth", map_location=f'cuda:{local_rank}', weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_valid_loss = ckpt["best_valid_loss"]
        best_loss_epoch = ckpt["best_loss_epoch"]
        train_losses = np.load(train_loss_file)
        valid_losses = np.load(valid_loss_file)

    ## Distributed model
    if cfg.world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    world_rank == 0 and logger.info(f"start training ...")

    for epoch in range(cfg.max_epoch):
        if dist.is_initialized():
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0
        train_batches = 0
        skipped_train_batches = 0
        start_time = time.time()
        for j, data in enumerate(train_dataloader):
            invar = data[0]
            t0_names = extract_t0_names(data[4], invar.shape[0])
            available = rollout_labels_available(
                t0_names,
                cfg_data.dataset.data_dir,
                train_years,
            )
            if not all_ranks_have_labels(available, local_rank):
                skipped_train_batches += 1
                continue

            optimizer.zero_grad()
            loss = autoregressive_train_loss(
                model=model,
                initial_state=invar,
                t0_names=t0_names,
                surface_mask=surface_mask,
                surface_weights=surface_weights,
                pressure_weights=pressure_weights,
                data_dir=cfg_data.dataset.data_dir,
                allowed_years=train_years,
                channel_indices=channel_indices,
                means=means,
                stds=stds,
                local_rank=local_rank,
                world_size=cfg.world_size,
            )
            optimizer.step()
            train_loss += loss
            train_batches += 1
            if world_rank == 0:
                logger.info(f'Train: Epoch {epoch}-{j+1}/{len(train_dataloader)} '
                            f'[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] '
                            f'[{(time.time()-start_time)/train_batches: .02f}s/{cfg_data.dataloader.batch_size}batch] '
                            f'loss:{train_loss / train_batches: .04f}')

        if train_batches == 0:
            raise RuntimeError("没有可用于8步自回归训练的样本")
        train_loss /= train_batches
        if world_rank == 0 and skipped_train_batches:
            logger.info(
                "Train skipped %d batches without complete +6h...+48h labels",
                skipped_train_batches,
            )

        model.eval()
        valid_loss = 0
        valid_batches = 0
        skipped_valid_batches = 0
        with torch.no_grad():
            start_time = time.time()
            for j, data in enumerate(val_dataloader):
                invar = data[0]
                t0_names = extract_t0_names(data[4], invar.shape[0])
                available = rollout_labels_available(
                    t0_names,
                    cfg_data.dataset.data_dir,
                    val_years,
                )
                if not all_ranks_have_labels(available, local_rank):
                    skipped_valid_batches += 1
                    continue

                loss = autoregressive_valid_loss(
                    model=model,
                    initial_state=invar,
                    t0_names=t0_names,
                    surface_mask=surface_mask,
                    surface_weights=surface_weights,
                    pressure_weights=pressure_weights,
                    data_dir=cfg_data.dataset.data_dir,
                    allowed_years=val_years,
                    channel_indices=channel_indices,
                    means=means,
                    stds=stds,
                    local_rank=local_rank,
                    world_size=cfg.world_size,
                )

                if cfg.world_size > 1:
                    loss_tensor = torch.tensor(loss, device=local_rank)
                    dist.all_reduce(loss_tensor)
                    loss = loss_tensor.item() / cfg.world_size
                valid_loss += loss
                valid_batches += 1
                if world_rank == 0:
                    logger.info(f'Valid: Epoch {epoch}-{j+1}/{len(val_dataloader)} '
                            f'[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] '
                            f'[{(time.time()-start_time)/valid_batches: .02f}s/{cfg_data.dataloader.batch_size}batch] '
                            f'loss:{valid_loss / valid_batches: .04f}')

        if valid_batches == 0:
            raise RuntimeError("没有可用于8步自回归验证的样本")
        valid_loss /= valid_batches
        if world_rank == 0 and skipped_valid_batches:
            logger.info(
                "Valid skipped %d batches without complete +6h...+48h labels",
                skipped_valid_batches,
            )
        is_save_ckp = False
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_loss_epoch = epoch
            world_rank == 0 and save_checkpoint(model, optimizer, scheduler, best_valid_loss, best_loss_epoch, cfg.checkpoint_dir)
            is_save_ckp = True

        scheduler.step()

        if world_rank == 0:
            logger.info(f"Epoch [{epoch}/{cfg.max_epoch}], "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Valid Loss: {valid_loss:.4f}, "
                        f"Best loss at Epoch: {best_loss_epoch}"
                        + (", saving checkpoint" if is_save_ckp else "")
                        )
            train_losses = np.append(train_losses, train_loss)
            valid_losses = np.append(valid_losses, valid_loss)

            np.save(train_loss_file, train_losses)
            np.save(valid_loss_file, valid_losses)

        if epoch - best_loss_epoch > cfg.patience:
            print(f"Loss has not decrease in {cfg.patience} epochs, stopping training...")
            exit()


def save_checkpoint(model, optimizer, scheduler, best_valid_loss, best_loss_epoch, model_path):
    model_to_save = model.module if hasattr(model, "module") else model
    state = {"model_state_dict": model_to_save.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
             "scheduler_state_dict": scheduler.state_dict(),
             "best_valid_loss": best_valid_loss,
             "best_loss_epoch": best_loss_epoch,
            }
    torch.save(state, f"{model_path}/model.pth")
    ### the weight file saving may interrupted due to DCU queue limit, get a backup to ensure there at least has one model 
    os.system(f"mv {model_path}/model.pth {model_path}/model_bak.pth")


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    main()
