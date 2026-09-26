#!/usr/bin/env python3
"""AI4S submission inference for the compact Patch-Lite 2x8x8 model.

Inference-only VRAM optimizations (no retrain, no new deps):
- FP16 + W8A16 weight-only checkpoint
- Blockwise INT4 Earth-position-bias storage with an INT8 runtime cache
- Blockwise W5 storage for deep Linear weights, unpacked once at model load
- Plain low-bit checkpoint whose full file size is measured
- Full 69-channel chunked patchrecovery3d
- Width-chunked shallow attention with in-place Q scaling/bias/softmax
- Split Q/K/V projection in deep layers for lower latency at the same peak
- Lean transformer blocks with early release and in-place GELU/residuals
- Lean Pangu.forward that drops input/activation refs before layer1 attention
- Early release of temporary tensors in the dataloader loop
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Iterable

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from onescience.datapipes.climate import ERA5Datapipe
from onescience.models.pangu import Pangu
from onescience.modules import OneRecovery
from onescience.modules.func_utils import crop3d, window_partition, window_reverse
from onescience.utils.YParams import YParams
from tqdm import tqdm
import gc

from quantize_w8a16 import replace_modules_with_w8a16


PATCH_SIZE = [2, 8, 8]
EMBED_DIM = 192
NUM_HEADS = [6, 12, 12, 6]
DTYPE = torch.float16
DEVICE = "cuda:0"
CONFIG_PATH = "conf/config.yaml"
CHECKPOINT_NAME = "w4_w8a16.pth"
RECOVERY3D_CHUNK_WIDTH = max(
    1,
    int(os.environ.get("PANGU_RECOVERY3D_CHUNK_WIDTH", "45")),
)
RECOVERY3D_MODE = os.environ.get(
    "PANGU_RECOVERY3D_MODE",
    "chunked",
).strip().lower()
if RECOVERY3D_MODE not in {"full", "chunked"}:
    raise ValueError("PANGU_RECOVERY3D_MODE must be one of full, chunked")
WARMUP_STEPS = max(0, int(os.environ.get("PANGU_WARMUP_STEPS", "0")))
OUTPUT_DIR = Path("result/output")
TIME_RECORD_PATH = Path("result/time_record.json")
MEMORY_RECORD_PATH = os.environ.get("PANGU_MEMORY_RECORD_PATH", "").strip()
ATTENTION_MEMORY_MODE = os.environ.get(
    "PANGU_ATTENTION_MEMORY_MODE", "hybrid_qscale"
).strip().lower()
LEAN_BLOCK_FORWARD = os.environ.get("PANGU_LEAN_BLOCK_FORWARD", "1") == "1"
ATTENTION_WIDTH_CHUNK = max(
    0,
    int(os.environ.get("PANGU_ATTENTION_WIDTH_CHUNK", "4")),
)
ATTENTION_CHUNK_DEEP = os.environ.get("PANGU_ATTENTION_CHUNK_DEEP", "0") == "1"
ATTENTION_CHUNK_STAGE = os.environ.get(
    "PANGU_ATTENTION_CHUNK_STAGE",
    "all",
).strip()
MLP_TOKEN_CHUNK = max(
    0,
    int(os.environ.get("PANGU_MLP_TOKEN_CHUNK", "43680")),
)
MLP_CHUNK_DEEP = os.environ.get("PANGU_MLP_CHUNK_DEEP", "0") == "1"
MLP_RESIDUAL_OUT = os.environ.get("PANGU_MLP_RESIDUAL_OUT", "1") == "1"
MLP_RESIDUAL_BIAS = os.environ.get(
    "PANGU_MLP_RESIDUAL_BIAS",
    "add",
).strip().lower()
if MLP_RESIDUAL_BIAS not in {"add", "drop"}:
    raise ValueError("PANGU_MLP_RESIDUAL_BIAS must be one of add, drop")
GELU_APPROXIMATE = os.environ.get(
    "PANGU_GELU_APPROXIMATE",
    "tanh",
).strip().lower()
if GELU_APPROXIMATE not in {"none", "tanh"}:
    raise ValueError("PANGU_GELU_APPROXIMATE must be one of none, tanh")
FOLD_NORM_AFFINE = os.environ.get("PANGU_FOLD_NORM_AFFINE", "1") == "1"
FOLD_RESAMPLE_NORM_AFFINE = (
    os.environ.get("PANGU_FOLD_RESAMPLE_NORM_AFFINE", "1") == "1"
)
FOLD_Q_SCALE = os.environ.get("PANGU_FOLD_Q_SCALE", "1") == "1"
OFFLOAD_SKIP_TOKENS = os.environ.get("PANGU_OFFLOAD_SKIP_TOKENS", "0") == "1"
LINEAR_CACHE_MODE = os.environ.get("PANGU_LINEAR_CACHE_MODE", "all").strip().lower()
REUSE_CHUNK_ATTENTION_PARAMETERS = (
    os.environ.get("PANGU_REUSE_CHUNK_ATTENTION_PARAMETERS", "1") == "1"
)
REUSE_CHUNK_EPB_BIAS = os.environ.get("PANGU_REUSE_CHUNK_EPB_BIAS", "1") == "1"
OUTPUT_BACKED_EPB_BIAS = os.environ.get(
    "PANGU_OUTPUT_BACKED_EPB_BIAS",
    "0",
) == "1"
EPB_CACHE_MODE = os.environ.get("PANGU_EPB_CACHE_MODE", "none").strip().lower()
if EPB_CACHE_MODE not in {"none", "shallow", "all"}:
    raise ValueError("PANGU_EPB_CACHE_MODE must be one of none, shallow, all")
ROLL_MODE = os.environ.get("PANGU_ROLL_MODE", "torch").strip().lower()
if ROLL_MODE not in {"torch", "index"}:
    raise ValueError("PANGU_ROLL_MODE must be one of torch, index")
CHUNK_PROJECTION_MODE = os.environ.get(
    "PANGU_CHUNK_PROJECTION_MODE",
    "chunk",
).strip().lower()
if CHUNK_PROJECTION_MODE not in {"chunk", "output", "full"}:
    raise ValueError(
        "PANGU_CHUNK_PROJECTION_MODE must be one of chunk, output, full"
    )
CHUNK_LINEAR_OUT = os.environ.get("PANGU_CHUNK_LINEAR_OUT", "1") == "1"
SHIFT_WINDOW_MODE = os.environ.get(
    "PANGU_SHIFT_WINDOW_MODE",
    "fused",
).strip().lower()
if SHIFT_WINDOW_MODE not in {"standard", "fused", "fused_all"}:
    raise ValueError(
        "PANGU_SHIFT_WINDOW_MODE must be one of standard, fused, fused_all"
    )
DIRECT_WINDOW_REVERSE = (
    os.environ.get("PANGU_DIRECT_WINDOW_REVERSE", "1") == "1"
)
DIRECT_WINDOW_FORWARD = (
    os.environ.get("PANGU_DIRECT_WINDOW_FORWARD", "1") == "1"
)
ATTENTION_KERNEL = os.environ.get(
    "PANGU_ATTENTION_KERNEL",
    "conventional",
).strip().lower()
if ATTENTION_KERNEL not in {"conventional", "sdpa"}:
    raise ValueError(
        "PANGU_ATTENTION_KERNEL must be one of conventional, sdpa"
    )
ATTENTION_BIASMASK_MODE = os.environ.get(
    "PANGU_ATTENTION_BIASMASK_MODE",
    "fused_gfx936",
).strip().lower()
if ATTENTION_BIASMASK_MODE not in {
    "eager",
    "fused_gfx936",
}:
    raise ValueError(
        "PANGU_ATTENTION_BIASMASK_MODE must be one of "
        "eager, fused_gfx936"
    )
ATTENTION_MASK_MODE = os.environ.get(
    "PANGU_ATTENTION_MASK_MODE",
    "int8_shared",
).strip().lower()
if ATTENTION_MASK_MODE not in {"native", "shared", "int8", "int8_shared"}:
    raise ValueError(
        "PANGU_ATTENTION_MASK_MODE must be one of "
        "native, shared, int8, int8_shared"
    )
LAYER_NORM_MODE = os.environ.get(
    "PANGU_LAYER_NORM_MODE",
    "gfx936",
).strip().lower()
if LAYER_NORM_MODE not in {"eager", "gfx936"}:
    raise ValueError("PANGU_LAYER_NORM_MODE must be one of eager, gfx936")
WINDOW_GATHER_MODE = os.environ.get(
    "PANGU_WINDOW_GATHER_MODE",
    "gfx936",
).strip().lower()
if WINDOW_GATHER_MODE not in {"eager", "gfx936"}:
    raise ValueError("PANGU_WINDOW_GATHER_MODE must be one of eager, gfx936")
FUSE_LAYER_NORM_GATHER = (
    os.environ.get("PANGU_FUSE_LAYER_NORM_GATHER", "1") == "1"
)
OUTPUT_AFFINE_MODE = os.environ.get(
    "PANGU_OUTPUT_AFFINE_MODE",
    "gfx936",
).strip().lower()
if OUTPUT_AFFINE_MODE not in {"eager", "gfx936"}:
    raise ValueError("PANGU_OUTPUT_AFFINE_MODE must be one of eager, gfx936")
_FUSED_BIAS_MASK_ADD = None
_GFX936_LAYER_NORM = None
_GFX936_RESIDUAL_ADD_LAYER_NORM = None
_GFX936_INDEX_SELECT_TOKENS = None
_GFX936_LAYER_NORM_GATHER = None
_GFX936_AFFINE_FP16_TO_FP32 = None
CONV_CACHE = os.environ.get("PANGU_CONV_CACHE", "1") == "1"
INPUT_ASSEMBLY_MODE = os.environ.get(
    "PANGU_INPUT_ASSEMBLY_MODE",
    "direct",
).strip().lower()
if INPUT_ASSEMBLY_MODE not in {"concat", "direct"}:
    raise ValueError("PANGU_INPUT_ASSEMBLY_MODE must be one of concat, direct")
SPLIT_MODEL_INPUT = os.environ.get("PANGU_SPLIT_MODEL_INPUT", "0") == "1"
STREAMED_MODEL_INPUT = os.environ.get("PANGU_STREAMED_MODEL_INPUT", "1") == "1"
if SPLIT_MODEL_INPUT and STREAMED_MODEL_INPUT:
    raise ValueError("split and streamed model input modes are mutually exclusive")
OUTPUT_TRANSFER_CHUNK = max(
    1,
    int(os.environ.get("PANGU_OUTPUT_TRANSFER_CHUNK", "8")),
)
OUTPUT_PIPELINE = os.environ.get("PANGU_OUTPUT_PIPELINE", "1") == "1"
OUTPUT_PIPELINE_BUFFERS = 2
OUTPUT_ASSEMBLY_MODE = os.environ.get(
    "PANGU_OUTPUT_ASSEMBLY", "cat"
).strip().lower()
if OUTPUT_ASSEMBLY_MODE not in {"cat", "split"}:
    raise ValueError("PANGU_OUTPUT_ASSEMBLY must be one of cat, split")
_OUTPUT_COPY_STREAM = None
_OUTPUT_READY_EVENTS = None
_OUTPUT_COPY_DONE_EVENTS = None
OUTPUT_DENORMALIZE_MODE = os.environ.get(
    "PANGU_OUTPUT_DENORMALIZE_MODE",
    "gpu",
).strip().lower()
if OUTPUT_DENORMALIZE_MODE not in {"cpu", "gpu"}:
    raise ValueError("PANGU_OUTPUT_DENORMALIZE_MODE must be one of cpu, gpu")
PIN_INPUT_MEMORY = os.environ.get("PANGU_PIN_INPUT_MEMORY", "0") == "1"
EMPTY_CACHE_BEFORE_INFERENCE = (
    os.environ.get("PANGU_EMPTY_CACHE_BEFORE_INFERENCE", "0") == "1"
)
EXECUTION_MODE = os.environ.get("PANGU_EXECUTION_MODE", "eager").strip().lower()
if EXECUTION_MODE not in {"eager", "cuda_graph"}:
    raise ValueError("PANGU_EXECUTION_MODE must be one of eager, cuda_graph")
if LINEAR_CACHE_MODE not in {
    "none",
    "shallow_attention",
    "attention",
    "mlp",
    "all",
}:
    raise ValueError(
        "PANGU_LINEAR_CACHE_MODE must be one of "
        "none, shallow_attention, attention, mlp, all"
    )
if ATTENTION_MEMORY_MODE not in {
    "baseline",
    "release",
    "inplace",
    "compact",
    "compact_inplace",
    "split_qkv_inplace",
    "hybrid",
    "qk_inplace",
    "hybrid_qk",
    "compact_inplace_qscale",
    "hybrid_qscale",
    "compact_qkv_inplace",
    "hybrid_compact_qkv",
}:
    raise ValueError(
        "PANGU_ATTENTION_MEMORY_MODE must be one of "
        "baseline, release, inplace, compact, compact_inplace, "
        "split_qkv_inplace, hybrid, qk_inplace, hybrid_qk, "
        "compact_inplace_qscale, hybrid_qscale, compact_qkv_inplace, "
        "hybrid_compact_qkv"
    )

FORECAST_STEPS = 8                # AI4S, 不可更改
FORECAST_INTERVAL_HOURS = 6.      # AI4S, 不可更改


def get_skip_sample_names(data_dir, test_years, skip_count):
    """返回每个测试年份最后 skip_count 个 dataloader 输入样本名。"""
    skip_names = set()

    for year in test_years:
        year_files = sorted(
            glob.glob(
                os.path.join(data_dir, "data", str(year), "*.h5")
            )
        )
        if not year_files:
            raise FileNotFoundError(
                f"未找到 {year} 年测试数据: "
                f"{os.path.join(data_dir, 'data', str(year))}"
            )

        # dataloader 需要同时构造 T0 输入和 T0+6h 标签，
        # 因此该年最后一个 HDF5 文件通常不会作为输入样本。
        input_sample_names = [
            os.path.splitext(os.path.basename(path))[0]
            for path in year_files[:-1]
        ]
        if len(input_sample_names) < skip_count:
            raise ValueError(
                f"{year} 年只有 {len(input_sample_names)} 个输入样本，"
                f"无法跳过最后 {skip_count} 个"
            )

        year_skip_names = input_sample_names[-skip_count:]
        skip_names.update(year_skip_names)
        print(
            f"{year}: input samples={len(input_sample_names)}, "
            f"skip final {len(year_skip_names)} samples"
        )
        print(f"{year} skipped samples: {year_skip_names}")

    return skip_names

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Accept both the organizer's no-argument entry and local score harness."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--checkpoint")
    parser.add_argument("--dtype", choices=("fp16",), default="fp16")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--time-record", default=str(TIME_RECORD_PATH))
    parser.add_argument("--memory-record", default=MEMORY_RECORD_PATH)
    parser.add_argument("--disable-torch-memory", action="store_true")
    parser.add_argument("--require-output", action="store_true")
    parser.add_argument("--official-timing", action="store_true")
    return parser.parse_args(argv)


def resolve_checkpoint_path(explicit: str | None, cfg: object) -> str:
    """Resolve the official checkpoint first, then a packaged sibling fallback."""
    if explicit:
        return explicit

    configured = Path(getattr(cfg, "checkpoint_dir")) / CHECKPOINT_NAME
    if configured.exists():
        return str(configured)

    sibling = Path.cwd().parent / "checkpoints" / CHECKPOINT_NAME
    if sibling.exists():
        return str(sibling)

    return str(configured)


def summarize_memory_samples(
    samples: list[dict[str, int]],
) -> dict[str, float | int | str]:
    if not samples:
        return {"source": "torch.cuda", "num_samples": 0}
    keys = tuple(samples[0])
    summary: dict[str, float | int | str] = {
        "source": "torch.cuda",
        "num_samples": len(samples),
    }
    for key in keys:
        values = [float(sample[key]) for sample in samples]
        metric = key.removesuffix("_bytes")
        summary[f"avg_{key}"] = sum(values) / len(values)
        summary[f"peak_{key}"] = max(values)
        summary[f"avg_{metric}_gb"] = summary[f"avg_{key}"] / 1e9
        summary[f"peak_{metric}_gb"] = summary[f"peak_{key}"] / 1e9
        summary[f"avg_{metric}_gib"] = summary[f"avg_{key}"] / 2**30
        summary[f"peak_{metric}_gib"] = summary[f"peak_{key}"] / 2**30
    return summary


class ChunkedPatchRecovery3D(nn.Module):
    """Recover every pressure level while bounding longitude workspace."""

    def __init__(self, base: nn.Module, chunk_width: int):
        super().__init__()
        self.base = base
        self.chunk_width = max(1, int(chunk_width))

    @property
    def recovery(self) -> nn.Module:
        return getattr(self.base, "recovery", self.base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            return self.base(x)

        recovery = self.recovery
        proj = recovery.proj
        img_size = tuple(recovery.img_size)
        patch_size = tuple(recovery.patch_size)
        if patch_size[2] <= 0:
            return self.base(x)

        batch = x.shape[0]
        out = x.new_empty((batch, recovery.out_chans, *img_size))

        for width_start in range(0, x.shape[-1], self.chunk_width):
            width_stop = min(width_start + self.chunk_width, x.shape[-1])
            out_start = width_start * patch_size[2]
            out_stop = width_stop * patch_size[2]
            chunk = proj(x[:, :, :, :, width_start:width_stop])
            pressure_pad = chunk.shape[2] - img_size[0]
            height_pad = chunk.shape[3] - img_size[1]
            if pressure_pad < 0 or height_pad < 0:
                raise ValueError("Recovered feature map is smaller than target img_size")
            pressure_front = pressure_pad // 2
            height_top = height_pad // 2
            out[:, :, :, :, out_start:out_stop] = chunk[
                :,
                :,
                pressure_front : pressure_front + img_size[0],
                height_top : height_top + img_size[1],
                : out_stop - out_start,
            ]
            del chunk
        return out


def lean_pangu_forward(self, x: torch.Tensor):
    """Drop large intermediates before layer1 attention sets the reserved peak."""
    streamed_input = isinstance(x, tuple) and len(x) == 3
    if streamed_input:
        invar, surface_mask, model_img_size = x
        surface = prepare_streamed_surface_input(
            invar,
            surface_mask,
            model_img_size,
            torch.device(DEVICE),
        )
    elif isinstance(x, tuple):
        surface, upper = x
    else:
        surface = x[:, :7].contiguous()
        upper = x[:, 7:].reshape(
            x.shape[0], 5, 13, x.shape[2], x.shape[3]
        ).contiguous()
        del x

    surface_features = self.patchembed2d(surface)
    del surface
    if streamed_input:
        upper = prepare_streamed_upper_input(
            invar,
            model_img_size,
            torch.device(DEVICE),
        )
        del invar, surface_mask, model_img_size
    upper_features = self.patchembed3d(upper)
    del upper

    combined = torch.concat([surface_features.unsqueeze(2), upper_features], dim=2)
    del surface_features, upper_features
    batch, channels, pressure_levels, height, width = combined.shape
    tokens = combined.reshape(batch, channels, -1).transpose(1, 2).contiguous()
    del combined

    tokens = self.layer1(tokens)
    if OFFLOAD_SKIP_TOKENS and tokens.is_cuda:
        skip_tokens_cpu = torch.empty_like(tokens, device="cpu", pin_memory=True)
        current_stream = torch.cuda.current_stream(tokens.device)
        self._skip_copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self._skip_copy_stream):
            skip_tokens_cpu.copy_(tokens, non_blocking=True)
            tokens.record_stream(self._skip_copy_stream)
    else:
        skip_tokens = tokens
    tokens = self.downsample(tokens)
    tokens = self.layer2(tokens)
    tokens = self.layer3(tokens)
    tokens = self.upsample(tokens)
    tokens = self.layer4(tokens)

    if OFFLOAD_SKIP_TOKENS and tokens.is_cuda:
        torch.cuda.current_stream(tokens.device).wait_stream(self._skip_copy_stream)
        skip_tokens = skip_tokens_cpu.to(device=tokens.device, non_blocking=True)
        del skip_tokens_cpu
    output_features = torch.concat([tokens, skip_tokens], dim=-1)
    del tokens, skip_tokens
    output_features = output_features.transpose(1, 2).reshape(
        batch, -1, pressure_levels, height, width
    )
    output_surface = self.patchrecovery2d(output_features[:, :, 0, :, :])
    output_upper_air = self.patchrecovery3d(output_features[:, :, 1:, :, :])
    del output_features
    return output_surface, output_upper_air


def lean_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Inference-only MLP that reuses the 4x hidden buffer for GELU."""
    chunk_size = MLP_TOKEN_CHUNK
    if (
        chunk_size <= 0
        or x.shape[1] <= chunk_size
        or (self.fc1.in_features > EMBED_DIM and not MLP_CHUNK_DEEP)
    ):
        x = self.fc1(x)
        apply_gelu_(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)

    output = torch.empty_like(x)
    for start in range(0, x.shape[1], chunk_size):
        stop = min(start + chunk_size, x.shape[1])
        hidden = self.fc1(x[:, start:stop])
        apply_gelu_(hidden)
        hidden = self.drop(hidden)
        chunk = self.drop(self.fc2(hidden))
        output[:, start:stop].copy_(chunk)
        del hidden, chunk
    return output


def apply_q_scale_(q: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply attention Q scaling, skipping the no-op after weight folding."""
    if scale != 1.0:
        q.mul_(scale)
    return q


def apply_gelu_(input: torch.Tensor) -> torch.Tensor:
    torch.ops.aten.gelu_(input, approximate=GELU_APPROXIMATE)
    return input


def mlp_add_residual_(
    mlp: nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor,
) -> None:
    """Evaluate the MLP directly into its residual destination."""
    weight, bias = dequantize_linear_parameters(mlp.fc2, x)
    chunk_size = MLP_TOKEN_CHUNK
    use_chunks = (
        chunk_size > 0
        and x.shape[1] > chunk_size
        and not (mlp.fc1.in_features > EMBED_DIM and not MLP_CHUNK_DEEP)
    )
    if not use_chunks:
        chunk_size = x.shape[1]
    for start in range(0, x.shape[1], chunk_size):
        stop = min(start + chunk_size, x.shape[1])
        hidden = mlp.fc1(x[:, start:stop])
        apply_gelu_(hidden)
        hidden = mlp.drop(hidden)
        target = residual[:, start:stop].reshape(-1, residual.shape[-1])
        if bias is not None and MLP_RESIDUAL_BIAS == "add":
            target.add_(bias)
        torch.addmm(
            target,
            hidden.reshape(-1, hidden.shape[-1]),
            weight.t(),
            beta=1,
            out=target,
        )
        del hidden, target
    del weight, bias


def indexed_roll_3d(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Apply a three-axis cyclic shift with one indexed copy."""
    batch, pressure, height, width, channels = x.shape
    flat = x.reshape(batch, pressure * height * width, channels)
    return torch.index_select(flat, 1, index).reshape(
        batch,
        pressure,
        height,
        width,
        channels,
    )


def install_roll_indices(model: nn.Module, device: torch.device) -> int:
    """Share precomputed fused-roll indices across equal-resolution blocks."""
    if ROLL_MODE != "index":
        return 0

    cache: dict[tuple[tuple[int, int, int], tuple[int, int, int]], torch.Tensor] = {}
    installed = 0
    for module in model.modules():
        if module.__class__.__name__ != "EarthTransformer3DBlock":
            continue
        if not module.use_roll:
            continue
        pressure, height, width = module.input_resolution
        left, right, top, bottom, front, back = module.pad.padding
        shape = (
            pressure + front + back,
            height + top + bottom,
            width + left + right,
        )
        shift = tuple(int(value) for value in module.shift_size)
        indices = []
        for signed_shift in (
            tuple(-value for value in shift),
            shift,
        ):
            key = (shape, signed_shift)
            if key not in cache:
                pressure_index = np.arange(shape[0], dtype=np.int64)[:, None, None]
                height_index = np.arange(shape[1], dtype=np.int64)[None, :, None]
                width_index = np.arange(shape[2], dtype=np.int64)[None, None, :]
                source_pressure = (pressure_index - signed_shift[0]) % shape[0]
                source_height = (height_index - signed_shift[1]) % shape[1]
                source_width = (width_index - signed_shift[2]) % shape[2]
                flat_index = (
                    source_pressure * (shape[1] * shape[2])
                    + source_height * shape[2]
                    + source_width
                )
                cache[key] = torch.from_numpy(
                    np.asarray(flat_index, dtype=np.int32).reshape(-1)
                ).to(device=device)
            indices.append(cache[key])
        module._roll_forward_index = indices[0]
        module._roll_reverse_index = indices[1]
        installed += 1
    return installed


def install_fused_window_indices(model: nn.Module, device: torch.device) -> int:
    """Precompute maps that fuse cyclic shift with partition/reverse copies."""
    if SHIFT_WINDOW_MODE not in {"fused", "fused_all"}:
        return 0

    cache: dict[
        tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
        tuple[torch.Tensor, torch.Tensor],
    ] = {}
    installed = 0
    for module in model.modules():
        if module.__class__.__name__ != "EarthTransformer3DBlock":
            continue
        if SHIFT_WINDOW_MODE == "fused" and not module.use_roll:
            continue
        pressure, height, width = module.input_resolution
        left, right, top, bottom, front, back = module.pad.padding
        shape = (
            pressure + front + back,
            height + top + bottom,
            width + left + right,
        )
        window = tuple(int(value) for value in module.window_size)
        shift = tuple(int(value) for value in module.shift_size)
        key = (shape, window, shift)
        if key not in cache:
            p_size, h_size, w_size = shape
            wp, wh, ww = window
            p_blocks = p_size // wp
            h_blocks = h_size // wh
            w_blocks = w_size // ww
            numel = p_size * h_size * w_size

            order = np.arange(numel, dtype=np.int64)
            remaining = order.copy()
            inner_w = remaining % ww
            remaining //= ww
            inner_h = remaining % wh
            remaining //= wh
            inner_p = remaining % wp
            remaining //= wp
            height_block = remaining % h_blocks
            remaining //= h_blocks
            pressure_block = remaining % p_blocks
            remaining //= p_blocks
            width_block = remaining % w_blocks
            shifted_p = pressure_block * wp + inner_p
            shifted_h = height_block * wh + inner_h
            shifted_w = width_block * ww + inner_w
            source_p = (shifted_p + shift[0]) % p_size
            source_h = (shifted_h + shift[1]) % h_size
            source_w = (shifted_w + shift[2]) % w_size
            forward = (
                source_p * (h_size * w_size)
                + source_h * w_size
                + source_w
            )

            spatial = np.arange(numel, dtype=np.int64)
            out_w = spatial % w_size
            out_h = (spatial // w_size) % h_size
            out_p = spatial // (h_size * w_size)
            source_p = (out_p - shift[0]) % p_size
            source_h = (out_h - shift[1]) % h_size
            source_w = (out_w - shift[2]) % w_size
            pressure_block, inner_p = np.divmod(source_p, wp)
            height_block, inner_h = np.divmod(source_h, wh)
            width_block, inner_w = np.divmod(source_w, ww)
            reverse = (
                (
                    (
                        (
                            (
                                width_block * p_blocks + pressure_block
                            )
                            * h_blocks
                            + height_block
                        )
                        * wp
                        + inner_p
                    )
                    * wh
                    + inner_h
                )
                * ww
                + inner_w
            )

            cache[key] = (
                torch.from_numpy(forward.astype(np.int32, copy=False)).to(
                    device=device
                ),
                torch.from_numpy(reverse.astype(np.int32, copy=False)).to(
                    device=device
                ),
            )
        module._fused_window_forward_index = cache[key][0]
        module._fused_window_reverse_index = cache[key][1]
        installed += 1
    return installed


def fused_shifted_window_partition(
    x: torch.Tensor,
    module: nn.Module,
) -> torch.Tensor:
    """Partition already-shifted windows with one indexed copy."""
    batch, pressure, height, width, channels = x.shape
    wp, wh, ww = module.window_size
    flat = x.reshape(batch, pressure * height * width, channels)
    windows = torch.index_select(
        flat,
        1,
        module._fused_window_forward_index,
    )
    return windows.reshape(
        batch * (width // ww),
        (pressure // wp) * (height // wh),
        wp,
        wh,
        ww,
        channels,
    )


def fused_shifted_window_reverse(
    windows: torch.Tensor,
    module: nn.Module,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    """Reverse window order and cyclic shift with one indexed copy."""
    pressure, height, width = shape
    width_windows = width // module.window_size[2]
    batch = windows.shape[0] // width_windows
    channels = windows.shape[-1]
    flat = windows.reshape(batch, pressure * height * width, channels)
    spatial = torch.index_select(
        flat,
        1,
        module._fused_window_reverse_index,
    )
    return spatial.reshape(batch, pressure, height, width, channels)


def install_direct_reverse_indices(model: nn.Module, device: torch.device) -> int:
    """Precompute direct maps between unpadded tokens and window order."""
    if not (DIRECT_WINDOW_FORWARD or DIRECT_WINDOW_REVERSE):
        return 0

    reverse_cache: dict[tuple, torch.Tensor] = {}
    forward_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
    installed = 0
    for module in model.modules():
        if module.__class__.__name__ != "EarthTransformer3DBlock":
            continue
        pressure, height, width = tuple(int(v) for v in module.input_resolution)
        left, right, top, bottom, front, back = (
            int(v) for v in module.pad.padding
        )
        padded_pressure = pressure + front + back
        padded_height = height + top + bottom
        padded_width = width + left + right
        window_pressure, window_height, window_width = (
            int(v) for v in module.window_size
        )
        shift = (
            tuple(int(v) for v in module.shift_size)
            if module.use_roll
            else (0, 0, 0)
        )
        key = (
            (pressure, height, width),
            (left, right, top, bottom, front, back),
            (window_pressure, window_height, window_width),
            shift,
        )
        if DIRECT_WINDOW_REVERSE and key not in reverse_cache:
            output = np.arange(pressure * height * width, dtype=np.int64)
            output_width = output % width + left
            output_height = (output // width) % height + top
            output_pressure = output // (height * width) + front
            source_pressure = (output_pressure - shift[0]) % padded_pressure
            source_height = (output_height - shift[1]) % padded_height
            source_width = (output_width - shift[2]) % padded_width
            pressure_block, inner_pressure = np.divmod(
                source_pressure,
                window_pressure,
            )
            height_block, inner_height = np.divmod(
                source_height,
                window_height,
            )
            width_block, inner_width = np.divmod(
                source_width,
                window_width,
            )
            pressure_blocks = padded_pressure // window_pressure
            height_blocks = padded_height // window_height
            direct = (
                (
                    (
                        (
                            (
                                width_block * pressure_blocks + pressure_block
                            )
                            * height_blocks
                            + height_block
                        )
                        * window_pressure
                        + inner_pressure
                    )
                    * window_height
                    + inner_height
                )
                * window_width
                + inner_width
            )
            reverse_cache[key] = torch.from_numpy(
                direct.astype(np.int32, copy=False)
            ).to(device=device)
        if DIRECT_WINDOW_REVERSE:
            module._direct_window_reverse_index = reverse_cache[key]

        if DIRECT_WINDOW_FORWARD and key not in forward_cache:
            order = np.arange(
                padded_pressure * padded_height * padded_width,
                dtype=np.int64,
            )
            remaining = order.copy()
            inner_width = remaining % window_width
            remaining //= window_width
            inner_height = remaining % window_height
            remaining //= window_height
            inner_pressure = remaining % window_pressure
            remaining //= window_pressure
            height_block = remaining % (
                padded_height // window_height
            )
            remaining //= padded_height // window_height
            pressure_block = remaining % (
                padded_pressure // window_pressure
            )
            remaining //= padded_pressure // window_pressure
            width_block = remaining
            shifted_pressure = (
                pressure_block * window_pressure + inner_pressure
            )
            shifted_height = height_block * window_height + inner_height
            shifted_width = width_block * window_width + inner_width
            source_pressure = (
                shifted_pressure + shift[0]
            ) % padded_pressure
            source_height = (shifted_height + shift[1]) % padded_height
            source_width = (shifted_width + shift[2]) % padded_width
            valid = (
                (source_pressure >= front)
                & (source_pressure < front + pressure)
                & (source_height >= top)
                & (source_height < top + height)
                & (source_width >= left)
                & (source_width < left + width)
            )
            source = (
                (source_pressure - front) * (height * width)
                + (source_height - top) * width
                + source_width
                - left
            )
            if WINDOW_GATHER_MODE == "gfx936":
                source[~valid] = -1
                invalid = np.empty(0, dtype=np.int64)
            else:
                source[~valid] = 0
                invalid = np.flatnonzero(~valid).astype(np.int64, copy=False)
            forward_cache[key] = (
                torch.from_numpy(
                    source.astype(np.int32, copy=False)
                ).to(device=device),
                torch.from_numpy(invalid).to(device=device),
            )
        if DIRECT_WINDOW_FORWARD:
            (
                module._direct_window_forward_index,
                module._direct_window_forward_invalid,
            ) = forward_cache[key]
        installed += 1
    return installed


def optimize_attention_masks(model: nn.Module) -> tuple[int, int, int]:
    """Losslessly compress fixed 0/-100 masks and share identical buffers."""
    if ATTENTION_MASK_MODE == "native":
        return 0, 0, 0

    optimized = 0
    original_bytes = 0
    shared_masks: dict[tuple[int, ...], list[torch.Tensor]] = {}
    for module in model.modules():
        mask = module._buffers.get("attn_mask")
        if not isinstance(mask, torch.Tensor):
            continue
        original_bytes += mask.numel() * mask.element_size()
        optimized_mask = mask
        if ATTENTION_MASK_MODE.startswith("int8"):
            rounded = mask.round()
            if not torch.equal(mask, rounded):
                raise RuntimeError("attention mask contains non-integral values")
            optimized_mask = rounded.to(torch.int8)
            valid = torch.logical_or(
                optimized_mask == 0,
                optimized_mask == -100,
            ).all()
            if not bool(valid.item()):
                raise RuntimeError("attention mask contains values other than 0/-100")

        if ATTENTION_MASK_MODE in {"shared", "int8_shared"}:
            key = tuple(optimized_mask.shape)
            for candidate in shared_masks.get(key, []):
                if torch.equal(candidate, optimized_mask):
                    optimized_mask = candidate
                    break
            else:
                shared_masks.setdefault(key, []).append(optimized_mask)
        module._buffers["attn_mask"] = optimized_mask
        optimized += 1

    unique_masks: dict[int, torch.Tensor] = {}
    for module in model.modules():
        mask = module._buffers.get("attn_mask")
        if isinstance(mask, torch.Tensor):
            unique_masks[mask.data_ptr()] = mask
    optimized_bytes = sum(
        mask.numel() * mask.element_size() for mask in unique_masks.values()
    )
    return optimized, original_bytes, optimized_bytes


def load_fused_bias_mask_add() -> None:
    """Compile/load the gfx936 extension before entering measured samples."""
    global _FUSED_BIAS_MASK_ADD
    global _GFX936_LAYER_NORM, _GFX936_RESIDUAL_ADD_LAYER_NORM
    global _GFX936_INDEX_SELECT_TOKENS
    global _GFX936_LAYER_NORM_GATHER, _GFX936_AFFINE_FP16_TO_FP32
    if (
        ATTENTION_BIASMASK_MODE == "eager"
        and LAYER_NORM_MODE == "eager"
        and WINDOW_GATHER_MODE == "eager"
        and OUTPUT_AFFINE_MODE == "eager"
    ) or _FUSED_BIAS_MASK_ADD is not None:
        return
    if ATTENTION_MASK_MODE != "int8_shared":
        raise ValueError(
            "fused_gfx936 bias+mask requires "
            "PANGU_ATTENTION_MASK_MODE=int8_shared"
        )
    if ATTENTION_KERNEL != "conventional":
        raise ValueError("fused_gfx936 bias+mask requires conventional attention")
    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx936")
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR",
        str(Path.cwd() / ".torch_extensions"),
    )
    from gfx936_biasmask.loader import (
        affine_fp16_to_fp32,
        fused_bias_mask_add_,
        index_select_tokens,
        layer_norm_no_affine,
        layer_norm_gather,
        load_extension,
        residual_add_layer_norm_,
    )

    load_extension(verbose=os.environ.get("PANGU_EXTENSION_VERBOSE", "0") == "1")
    _FUSED_BIAS_MASK_ADD = fused_bias_mask_add_
    _GFX936_LAYER_NORM = layer_norm_no_affine
    _GFX936_RESIDUAL_ADD_LAYER_NORM = residual_add_layer_norm_
    _GFX936_INDEX_SELECT_TOKENS = index_select_tokens
    _GFX936_LAYER_NORM_GATHER = layer_norm_gather
    _GFX936_AFFINE_FP16_TO_FP32 = affine_fp16_to_fp32


def gfx936_layer_norm_forward(self, input: torch.Tensor) -> torch.Tensor:
    if _GFX936_LAYER_NORM is None:
        raise RuntimeError("gfx936 layer norm extension was not loaded")
    return _GFX936_LAYER_NORM(input, float(self.eps))


def gfx936_residual_add_layer_norm_(
    residual_destination: torch.Tensor,
    residual: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    if _GFX936_RESIDUAL_ADD_LAYER_NORM is None:
        raise RuntimeError("gfx936 residual LayerNorm extension was not loaded")
    return _GFX936_RESIDUAL_ADD_LAYER_NORM(
        residual_destination,
        residual,
        float(epsilon),
    )


def gfx936_index_select_tokens(
    input: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    if _GFX936_INDEX_SELECT_TOKENS is None:
        load_fused_bias_mask_add()
    if _GFX936_INDEX_SELECT_TOKENS is None:
        raise RuntimeError("gfx936 token gather extension was not loaded")
    return _GFX936_INDEX_SELECT_TOKENS(input, index)


def gfx936_layer_norm_gather(
    input: torch.Tensor,
    index: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    if _GFX936_LAYER_NORM_GATHER is None:
        load_fused_bias_mask_add()
    if _GFX936_LAYER_NORM_GATHER is None:
        raise RuntimeError("gfx936 LayerNorm gather extension was not loaded")
    return _GFX936_LAYER_NORM_GATHER(input, index, float(epsilon))


def gfx936_affine_fp16_to_fp32(
    input: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    if _GFX936_AFFINE_FP16_TO_FP32 is None:
        load_fused_bias_mask_add()
    if _GFX936_AFFINE_FP16_TO_FP32 is None:
        raise RuntimeError("gfx936 output affine extension was not loaded")
    return _GFX936_AFFINE_FP16_TO_FP32(input, mean, std, output)


def install_gfx936_layer_norm(model: nn.Module) -> int:
    if LAYER_NORM_MODE != "gfx936":
        return 0
    load_fused_bias_mask_add()
    replaced = 0
    for module in model.modules():
        if not isinstance(module, nn.LayerNorm):
            continue
        if module.weight is not None or module.bias is not None:
            continue
        normalized_shape = tuple(module.normalized_shape)
        if normalized_shape not in {(192,), (384,), (768,)}:
            continue
        module.forward = MethodType(gfx936_layer_norm_forward, module)
        replaced += 1
    return replaced


def apply_cached_bias_mask_(
    attn: torch.Tensor,
    earth_position_bias: torch.Tensor,
    mask: torch.Tensor | None,
) -> bool:
    """Apply EPB and the fixed shifted-window mask to attention in-place."""
    if ATTENTION_BIASMASK_MODE.startswith("fused_") and mask is not None:
        load_fused_bias_mask_add()
        if _FUSED_BIAS_MASK_ADD is None:
            raise RuntimeError("gfx936 fused bias+mask extension was not loaded")
        _FUSED_BIAS_MASK_ADD(attn, earth_position_bias, mask)
        return False
    attn.add_(earth_position_bias.unsqueeze(0))
    if mask is None:
        return False
    num_width_windows = mask.shape[0]
    original_shape = attn.shape
    viewed = attn.view(
        original_shape[0] // num_width_windows,
        num_width_windows,
        *original_shape[1:],
    )
    viewed.add_(mask.unsqueeze(1).unsqueeze(0))
    return False


def direct_window_reverse_crop(
    windows: torch.Tensor,
    module: nn.Module,
) -> torch.Tensor:
    """Reverse windows, cyclic shift, and padding crop with one indexed copy."""
    padded_width = (
        module.input_resolution[2]
        + module.pad.padding[0]
        + module.pad.padding[1]
    )
    width_windows = padded_width // module.window_size[2]
    batch = windows.shape[0] // width_windows
    channels = windows.shape[-1]
    flat = windows.reshape(batch, -1, channels)
    if WINDOW_GATHER_MODE == "gfx936":
        return gfx936_index_select_tokens(
            flat,
            module._direct_window_reverse_index,
        )
    return torch.index_select(flat, 1, module._direct_window_reverse_index)


def direct_window_partition(
    x: torch.Tensor,
    module: nn.Module,
) -> torch.Tensor:
    """Pad, cyclically shift, and partition normalized tokens in one gather."""
    batch, _tokens, channels = x.shape
    pressure, height, width = module.input_resolution
    left, right, top, bottom, front, back = module.pad.padding
    padded_pressure = pressure + front + back
    padded_height = height + top + bottom
    padded_width = width + left + right
    window_pressure, window_height, window_width = module.window_size
    if WINDOW_GATHER_MODE == "gfx936":
        windows = gfx936_index_select_tokens(
            x,
            module._direct_window_forward_index,
        )
    else:
        windows = torch.index_select(
            x,
            1,
            module._direct_window_forward_index,
        )
    if module._direct_window_forward_invalid.numel():
        windows.index_fill_(
            1,
            module._direct_window_forward_invalid,
            0,
        )
    return windows.reshape(
        batch * (padded_width // window_width),
        (padded_pressure // window_pressure)
        * (padded_height // window_height),
        window_pressure,
        window_height,
        window_width,
        channels,
    )


def direct_window_partition_layer_norm(
    x: torch.Tensor,
    module: nn.Module,
    epsilon: float,
) -> torch.Tensor:
    """Fuse norm1 with the direct padded/shifted window gather."""
    batch, _tokens, channels = x.shape
    pressure, height, width = module.input_resolution
    left, right, top, bottom, front, back = module.pad.padding
    padded_pressure = pressure + front + back
    padded_height = height + top + bottom
    padded_width = width + left + right
    window_pressure, window_height, window_width = module.window_size
    windows = gfx936_layer_norm_gather(
        x,
        module._direct_window_forward_index,
        epsilon,
    )
    return windows.reshape(
        batch * (padded_width // window_width),
        (padded_pressure // window_pressure) * (padded_height // window_height),
        window_pressure,
        window_height,
        window_width,
        channels,
    )


def lean_earth_transformer_forward(self, x: torch.Tensor) -> torch.Tensor:
    """EarthTransformer3DBlock forward with early buffer release."""
    pressure_levels, height, width = self.input_resolution
    batch, _num_tokens, channels = x.shape

    shortcut = x
    left, right, top, bottom, front, back = self.pad.padding
    padded_pressure = pressure_levels + front + back
    padded_height = height + top + bottom
    padded_width = width + left + right

    shift_pressure, shift_height, shift_width = self.shift_size
    fused_windows = SHIFT_WINDOW_MODE == "fused_all" or (
        self.use_roll and SHIFT_WINDOW_MODE == "fused"
    )
    if DIRECT_WINDOW_FORWARD:
        if (
            FUSE_LAYER_NORM_GATHER
            and LAYER_NORM_MODE == "gfx936"
            and WINDOW_GATHER_MODE == "gfx936"
        ):
            windows = direct_window_partition_layer_norm(
                x,
                self,
                self.norm1.eps,
            )
        else:
            normalized = self.norm1(x)
            windows = direct_window_partition(normalized, self)
            del normalized
    else:
        padded = self.norm1(x).view(
            batch,
            pressure_levels,
            height,
            width,
            channels,
        )
        padded = self.pad(
            padded.permute(0, 4, 1, 2, 3)
        ).permute(0, 2, 3, 4, 1)
        if fused_windows:
            windows = fused_shifted_window_partition(padded, self)
            del padded
        elif self.use_roll:
            if ROLL_MODE == "index":
                shifted = indexed_roll_3d(padded, self._roll_forward_index)
            else:
                shifted = torch.roll(
                    padded,
                    shifts=(-shift_pressure, -shift_height, -shift_width),
                    dims=(1, 2, 3),
                )
            del padded
        else:
            shifted = padded
            del padded

        if not fused_windows:
            windows = window_partition(shifted, self.window_size)
            del shifted
    window_pressure, window_height, window_width = self.window_size
    windows = windows.view(
        windows.shape[0],
        windows.shape[1],
        window_pressure * window_height * window_width,
        channels,
    )
    attended = self.attn(windows, mask=self.attn_mask)
    del windows
    attended = attended.view(
        attended.shape[0],
        attended.shape[1],
        window_pressure,
        window_height,
        window_width,
        channels,
    )

    if DIRECT_WINDOW_REVERSE:
        residual = direct_window_reverse_crop(attended, self)
        del attended
    elif fused_windows:
        padded = fused_shifted_window_reverse(
            attended,
            self,
            (padded_pressure, padded_height, padded_width),
        )
        del attended
    else:
        shifted = window_reverse(
            attended,
            self.window_size,
            Pl=padded_pressure,
            Lat=padded_height,
            Lon=padded_width,
        )
        del attended
    if DIRECT_WINDOW_REVERSE:
        pass
    elif self.use_roll and not fused_windows:
        if ROLL_MODE == "index":
            padded = indexed_roll_3d(shifted, self._roll_reverse_index)
        else:
            padded = torch.roll(
                shifted,
                shifts=(shift_pressure, shift_height, shift_width),
                dims=(1, 2, 3),
            )
        del shifted
    elif not self.use_roll and not fused_windows:
        padded = shifted
        del shifted

    if not DIRECT_WINDOW_REVERSE:
        residual = crop3d(
            padded.permute(0, 4, 1, 2, 3),
            self.input_resolution,
        ).permute(0, 2, 3, 4, 1)
        del padded
        residual = residual.reshape(
            batch,
            pressure_levels * height * width,
            channels,
        )
    fused_norm2 = None
    if LAYER_NORM_MODE == "gfx936" and MLP_RESIDUAL_OUT and not self.training:
        fused_norm2 = gfx936_residual_add_layer_norm_(
            shortcut,
            self.drop_path(residual),
            self.norm2.eps,
        )
    else:
        shortcut.add_(self.drop_path(residual))
    del residual

    if MLP_RESIDUAL_OUT and not self.training:
        normalized = fused_norm2
        if normalized is None:
            normalized = self.norm2(shortcut)
        mlp_add_residual_(self.mlp, normalized, shortcut)
        del normalized
    else:
        residual = self.mlp(self.norm2(shortcut))
        shortcut.add_(self.drop_path(residual))
    return shortcut


def get_stats(data_dir: str, channels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    metadata_path = os.path.join(data_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        all_variables = metadata["variables"]
    else:
        h5_files = sorted(glob.glob(os.path.join(data_dir, "data", "*.h5")))
        if not h5_files:
            raise FileNotFoundError(f"Cannot find metadata.json or data/*.h5 under {data_dir}")
        with h5py.File(h5_files[0], "r") as f:
            variables = f["fields"].attrs["variables"]
            all_variables = [
                value.decode() if isinstance(value, bytes) else value
                for value in variables
            ]

    channel_indices = [all_variables.index(v) for v in channels]
    stats_dir = os.path.join(data_dir, "stats")
    mu = np.load(os.path.join(stats_dir, "global_means.npy"))
    std = np.load(os.path.join(stats_dir, "global_stds.npy"))
    return mu[:, channel_indices, :, :], std[:, channel_indices, :, :]


def padded_img_size(img_size: Iterable[int], patch_size: Iterable[int]) -> list[int]:
    h, w = list(img_size)
    _p_pl, p_h, p_w = list(patch_size)
    return [((h + p_h - 1) // p_h) * p_h, ((w + p_w - 1) // p_w) * p_w]


def symmetric_padding(
    current_size: Iterable[int],
    target_size: Iterable[int],
) -> tuple[int, int, int, int]:
    h, w = list(current_size)
    target_h, target_w = list(target_size)
    pad_h = target_h - h
    pad_w = target_w - w
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"input shape {(h, w)} is larger than target img_size {list(target_size)}")
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return left, right, top, bottom


def pad_to_img_size(x: torch.Tensor, img_size: Iterable[int]) -> torch.Tensor:
    h, w = x.shape[-2:]
    left, right, top, bottom = symmetric_padding((h, w), img_size)
    if left == right == top == bottom == 0:
        return x
    return F.pad(x, (left, right, top, bottom), mode="replicate")


def crop_to_img_size(x: torch.Tensor, img_size: Iterable[int]) -> torch.Tensor:
    target_h, target_w = list(img_size)
    h, w = x.shape[-2:]
    left, _right, top, _bottom = symmetric_padding(img_size, (h, w))
    return x[..., top : top + target_h, left : left + target_w]


def load_serialized_checkpoint(path: Path) -> object:
    """Load a regular PyTorch checkpoint without nested XZ decompression."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint_state(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    ckpt = load_serialized_checkpoint(path)

    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    drop_patterns = (
        ".attn_mask",
        ".earth_position_index",
        "device_buffer",
        ".Fuser.",
        ".Sampler.",
        ".Reconvery.",
    )
    return {
        key: value
        for key, value in state.items()
        if not any(pattern in key for pattern in drop_patterns)
    }


def is_w8a16_state(state: dict[str, torch.Tensor]) -> bool:
    return any(
        key.endswith(".qweight")
        or key.endswith(".qweight_packed")
        or key.endswith(".qweight_packed5")
        for key in state
    )


def blockwise_w4_linear_forward(self, x: torch.Tensor) -> torch.Tensor:
    """W8A16 Linear compute using a blockwise-W4-derived INT8 runtime cache."""
    dtype = x.dtype if x.is_floating_point() else torch.float16
    out_features, in_features = self.qweight.shape
    num_blocks = self.scale.shape[1]
    block_size = in_features // num_blocks
    weight = self.qweight.to(device=x.device, dtype=dtype).reshape(
        out_features, num_blocks, block_size
    )
    weight = weight.mul(
        self.scale.to(device=x.device, dtype=dtype).unsqueeze(2)
    ).reshape(out_features, in_features)
    bias = None if self.bias is None else self.bias.to(device=x.device, dtype=dtype)
    return F.linear(x, weight, bias)


def unpack_w4_linear_state(
    model: nn.Module,
    state: dict[str, torch.Tensor],
) -> int:
    """Unpack blockwise W4 checkpoint tensors into INT8 runtime buffers."""
    modules = dict(model.named_modules())
    packed_specs = []
    for key in state:
        if "earth_position_bias_" in key:
            continue
        if key.endswith(".qweight_packed"):
            packed_specs.append((key, 4, ".qweight_packed"))
        elif key.endswith(".qweight_packed5"):
            packed_specs.append((key, 5, ".qweight_packed5"))
    for packed_key, bits, suffix in packed_specs:
        prefix = packed_key[: -len(suffix)]
        scale_key = f"{prefix}.scale"
        module = modules.get(prefix)
        if module is None or not hasattr(module, "qweight"):
            raise KeyError(f"cannot resolve packed W4 module: {prefix}")
        if scale_key not in state:
            raise KeyError(f"missing blockwise W4 scale: {scale_key}")

        packed = state.pop(packed_key)
        original_shape = tuple(module.qweight.shape)
        if bits == 4:
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            quantized = (
                torch.stack((low, high), dim=-1)
                .reshape(original_shape)
                .to(torch.int8)
                .sub_(8)
                .contiguous()
            )
        else:
            groups = packed.reshape(*packed.shape[:-1], -1, 5).to(torch.int64)
            word = torch.zeros(groups.shape[:-1], dtype=torch.int64)
            for index in range(5):
                word |= groups[..., index] << (8 * index)
            quantized = torch.stack(
                [(word >> (5 * index)) & 0x1F for index in range(8)],
                dim=-1,
            )
            quantized = (
                quantized.reshape(original_shape)
                .to(torch.int8)
                .sub_(16)
                .contiguous()
            )
        state[f"{prefix}.qweight"] = quantized
        module._buffers["scale"] = torch.empty_like(state[scale_key])
        module.forward = MethodType(blockwise_w4_linear_forward, module)
    return len(packed_specs)


def is_quantized_epb_state(state: dict[str, torch.Tensor]) -> bool:
    return any(
        key.endswith(".earth_position_bias_qweight")
        or key.endswith(".earth_position_bias_qweight_packed")
        for key in state
    )


def dequantize_epb_table(module: nn.Module, dtype: torch.dtype) -> torch.Tensor:
    """Restore one blockwise-quantized EPB table for the current attention call."""
    if hasattr(module, "earth_position_bias_table_fp16"):
        return module.earth_position_bias_table_fp16
    rows, pressure_height, heads = module._epb_original_shape
    block_size = module._epb_block_size
    scale = module.earth_position_bias_scale.to(dtype=dtype).unsqueeze(1)

    if hasattr(module, "earth_position_bias_qweight"):
        quantized = module.earth_position_bias_qweight.to(dtype=dtype)
    else:
        packed = module.earth_position_bias_qweight_packed
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        quantized = (
            torch.stack((low, high), dim=1)
            .reshape(rows, pressure_height, heads)
            .to(dtype=dtype)
            .sub_(8)
        )

    table = quantized.reshape(
        rows // block_size, block_size, pressure_height, heads
    )
    table = table.mul(scale).reshape(rows, pressure_height, heads)
    return table


def cache_epb_tables(model: nn.Module, mode: str) -> tuple[int, int]:
    """Cache selected EPB tables as FP16 and release their INT8 buffers."""
    if mode == "none":
        return 0, 0

    cached = 0
    added_bytes = 0
    for module in model.modules():
        if "earth_position_bias_qweight" not in module._buffers:
            continue
        if mode == "shallow" and module.qkv.in_features > EMBED_DIM:
            continue
        old_bytes = (
            module.earth_position_bias_qweight.numel()
            * module.earth_position_bias_qweight.element_size()
            + module.earth_position_bias_scale.numel()
            * module.earth_position_bias_scale.element_size()
        )
        table = dequantize_epb_table(module, DTYPE).contiguous()
        del module._buffers["earth_position_bias_qweight"]
        del module._buffers["earth_position_bias_scale"]
        module.register_buffer(
            "earth_position_bias_table_fp16",
            table,
            persistent=False,
        )
        added_bytes += table.numel() * table.element_size() - old_bytes
        cached += 1
    return cached, added_bytes


def dequantize_linear_parameters(
    module: nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Match W8/W5 Linear dequantization while exposing weight slices."""
    dtype = x.dtype if x.is_floating_point() else torch.float16
    if hasattr(module, "weight_fp16"):
        weight = module.weight_fp16
        bias = None if module.bias is None else module.bias.to(
            device=x.device,
            dtype=dtype,
        )
        return weight, bias
    if hasattr(module, "weight"):
        weight = module.weight.to(device=x.device, dtype=dtype)
        bias = None if module.bias is None else module.bias.to(
            device=x.device,
            dtype=dtype,
        )
        return weight, bias
    out_features, in_features = module.qweight.shape
    scale = module.scale.to(device=x.device, dtype=dtype)
    quantized = module.qweight.to(device=x.device, dtype=dtype)
    if scale.ndim == 1:
        weight = quantized.mul(scale[:, None])
    else:
        num_blocks = scale.shape[1]
        block_size = in_features // num_blocks
        weight = quantized.reshape(
            out_features, num_blocks, block_size
        ).mul(scale.unsqueeze(2)).reshape(out_features, in_features)
    bias = None if module.bias is None else module.bias.to(
        device=x.device, dtype=dtype
    )
    return weight, bias


def cached_linear_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Linear forward backed by a load-time FP16 weight cache."""
    return F.linear(x, self.weight_fp16, self.bias)


def cache_linear_weights(model: nn.Module, mode: str) -> tuple[int, int]:
    """Trade a small amount of resident VRAM for lower inference latency."""
    if mode == "none":
        return 0, 0

    cached = 0
    added_bytes = 0
    for module_name, module in model.named_modules():
        if not hasattr(module, "qweight") or module.qweight.ndim != 2:
            continue
        is_attention = ".attn." in module_name
        is_mlp = ".mlp." in module_name
        selected = (
            mode == "all"
            or (mode == "attention" and is_attention)
            or (mode == "mlp" and is_mlp)
            or (
                mode == "shallow_attention"
                and is_attention
                and module.in_features <= EMBED_DIM
            )
        )
        if not selected:
            continue

        old_bytes = (
            module.qweight.numel() * module.qweight.element_size()
            + module.scale.numel() * module.scale.element_size()
        )
        out_features, in_features = module.qweight.shape
        scale = module.scale.to(dtype=DTYPE)
        quantized = module.qweight.to(dtype=DTYPE)
        if scale.ndim == 1:
            weight = quantized.mul(scale[:, None])
        else:
            num_blocks = scale.shape[1]
            block_size = in_features // num_blocks
            weight = quantized.reshape(
                out_features,
                num_blocks,
                block_size,
            ).mul(scale.unsqueeze(2)).reshape(out_features, in_features)
        weight = weight.contiguous()
        del module._buffers["qweight"]
        del module._buffers["scale"]
        module.register_buffer("weight_fp16", weight, persistent=False)
        module.forward = MethodType(cached_linear_forward, module)
        added_bytes += weight.numel() * weight.element_size() - old_bytes
        cached += 1
    return cached, added_bytes


def cached_conv3d_forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.padding_mode != "zeros":
        x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        padding = 0
    else:
        padding = self.padding
    return F.conv3d(
        x,
        self.weight_fp16,
        self.bias,
        self.stride,
        padding,
        self.dilation,
        self.groups,
    )


def cached_conv_transpose3d_forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.conv_transpose3d(
        x,
        self.weight_fp16,
        self.bias,
        self.stride,
        self.padding,
        self.output_padding,
        self.groups,
        self.dilation,
    )


def cache_convolution_weights(model: nn.Module) -> tuple[int, int]:
    """Cache W8 convolution weights as FP16 for repeated recovery calls."""
    if not CONV_CACHE:
        return 0, 0

    cached = 0
    added_bytes = 0
    for module in model.modules():
        if not hasattr(module, "qweight") or module.qweight.ndim <= 2:
            continue
        old_bytes = (
            module.qweight.numel() * module.qweight.element_size()
            + module.scale.numel() * module.scale.element_size()
        )
        shape = [1] * module.qweight.ndim
        shape[0] = -1
        weight = module.qweight.to(dtype=DTYPE).mul(
            module.scale.to(dtype=DTYPE).reshape(shape)
        ).contiguous()
        del module._buffers["qweight"]
        del module._buffers["scale"]
        module.register_buffer("weight_fp16", weight, persistent=False)
        if hasattr(module, "output_padding"):
            module.forward = MethodType(cached_conv_transpose3d_forward, module)
        else:
            module.forward = MethodType(cached_conv3d_forward, module)
        added_bytes += weight.numel() * weight.element_size() - old_bytes
        cached += 1
    return cached, added_bytes


@torch.no_grad()
def fold_layernorm_affine(model: nn.Module) -> int:
    """Fold each pre-norm affine transform into its sole following Linear."""
    if not FOLD_NORM_AFFINE and not FOLD_RESAMPLE_NORM_AFFINE:
        return 0

    def fold_pair(norm: nn.LayerNorm, linear: nn.Module) -> bool:
        if norm.weight is None or not hasattr(linear, "weight_fp16"):
            return False
        weight = linear.weight_fp16.detach().cpu().float()
        gamma = norm.weight.detach().cpu().float()
        beta = norm.bias.detach().cpu().float()
        if linear.bias is None:
            bias = torch.zeros(weight.shape[0], dtype=torch.float32)
        else:
            bias = linear.bias.detach().cpu().float()
        folded_weight = (weight * gamma.unsqueeze(0)).to(DTYPE)
        folded_bias = F.linear(beta, weight, bias).to(DTYPE)
        linear.weight_fp16.copy_(folded_weight.to(device=linear.weight_fp16.device))
        if linear.bias is None:
            linear.bias = nn.Parameter(
                folded_bias.to(device=linear.weight_fp16.device),
                requires_grad=False,
            )
        else:
            linear.bias.copy_(folded_bias.to(device=linear.bias.device))
        norm.register_parameter("weight", None)
        norm.register_parameter("bias", None)
        return True

    folded = 0
    if FOLD_NORM_AFFINE:
        for block in model.modules():
            if block.__class__.__name__ != "EarthTransformer3DBlock":
                continue
            for norm, linear in (
                (block.norm1, block.attn.qkv),
                (block.norm2, block.mlp.fc1),
            ):
                folded += int(fold_pair(norm, linear))

    if FOLD_RESAMPLE_NORM_AFFINE:
        for norm, linear in (
            (model.downsample.sampler.norm, model.downsample.sampler.linear),
            (model.upsample.sampler.norm, model.upsample.sampler.linear2),
        ):
            folded += int(fold_pair(norm, linear))
    return folded


@torch.no_grad()
def fold_attention_q_scale(model: nn.Module) -> int:
    """Absorb 1/sqrt(d) into Q projection weights so runtime skips q.mul_."""
    if not FOLD_Q_SCALE:
        return 0

    folded = 0
    for module in model.modules():
        if module.__class__.__name__ != "EarthAttention3D":
            continue
        qkv = module.qkv
        if not hasattr(qkv, "weight_fp16"):
            continue
        channels = qkv.weight_fp16.shape[1]
        scale = float(module.scale)
        qkv.weight_fp16[:channels].mul_(scale)
        if qkv.bias is not None:
            qkv.bias[:channels].mul_(scale)
        module.scale = 1.0
        folded += 1
    return folded


def quantized_epb_attention_forward_full(self, x: torch.Tensor, mask=None):
    """EarthAttention3D forward with transient EPB dequantization."""
    batch_width, num_pressure_height, window_tokens, channels = x.shape
    deep_layer = channels > EMBED_DIM
    split_qkv = ATTENTION_MEMORY_MODE == "split_qkv_inplace" or (
        ATTENTION_MEMORY_MODE
        in {"hybrid", "hybrid_qk", "hybrid_qscale", "hybrid_compact_qkv"}
        and deep_layer
    )
    split_qk = ATTENTION_MEMORY_MODE == "qk_inplace" or (
        ATTENTION_MEMORY_MODE == "hybrid_qk" and not deep_layer
    )
    deferred_v = split_qkv or split_qk
    use_inplace = ATTENTION_MEMORY_MODE in {
        "inplace",
        "compact_inplace",
        "split_qkv_inplace",
        "hybrid",
        "qk_inplace",
        "hybrid_qk",
        "compact_inplace_qscale",
        "hybrid_qscale",
        "compact_qkv_inplace",
        "hybrid_compact_qkv",
    }
    compact_v = ATTENTION_MEMORY_MODE in {
        "compact",
        "compact_inplace",
        "compact_inplace_qscale",
    } or (
        ATTENTION_MEMORY_MODE in {"hybrid", "hybrid_qscale"} and not split_qkv
    )
    inplace_q_scale = ATTENTION_MEMORY_MODE in {
        "compact_inplace_qscale",
        "hybrid_qscale",
        "compact_qkv_inplace",
        "hybrid_compact_qkv",
    }
    compact_all_qkv = ATTENTION_MEMORY_MODE == "compact_qkv_inplace" or (
        ATTENTION_MEMORY_MODE == "hybrid_compact_qkv" and not split_qkv
    )
    head_dim = channels // self.num_heads
    if split_qkv:
        qkv_weight, qkv_bias = dequantize_linear_parameters(self.qkv, x)
        q = F.linear(
            x,
            qkv_weight[:channels],
            None if qkv_bias is None else qkv_bias[:channels],
        )
        apply_q_scale_(q, self.scale)
        q = q.reshape(
            batch_width,
            num_pressure_height,
            window_tokens,
            self.num_heads,
            head_dim,
        ).permute(0, 3, 1, 2, 4)
        k = F.linear(
            x,
            qkv_weight[channels : 2 * channels],
            None if qkv_bias is None else qkv_bias[channels : 2 * channels],
        )
        k = k.reshape(
            batch_width,
            num_pressure_height,
            window_tokens,
            self.num_heads,
            head_dim,
        ).permute(0, 3, 1, 2, 4)
        attn = q @ k.transpose(-2, -1)
        del q, k
    elif split_qk:
        qkv_weight, qkv_bias = dequantize_linear_parameters(self.qkv, x)
        qk = F.linear(
            x,
            qkv_weight[: 2 * channels],
            None if qkv_bias is None else qkv_bias[: 2 * channels],
        )
        qk = qk.reshape(
            batch_width,
            num_pressure_height,
            window_tokens,
            2,
            self.num_heads,
            head_dim,
        ).permute(3, 0, 4, 1, 2, 5)
        q, k = qk[0], qk[1]
        apply_q_scale_(q, self.scale)
        attn = q @ k.transpose(-2, -1)
        del q, k, qk
    else:
        qkv = (
            self.qkv(x)
            .reshape(
                batch_width,
                num_pressure_height,
                window_tokens,
                3,
                self.num_heads,
                head_dim,
            )
            .permute(3, 0, 4, 1, 2, 5)
        )
        if compact_all_qkv:
            qkv = qkv.contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        if inplace_q_scale:
            apply_q_scale_(q, self.scale)
        else:
            q = q * self.scale if self.scale != 1.0 else q
        attn = q @ k.transpose(-2, -1)
        if compact_v:
            v = v.contiguous()
        if ATTENTION_MEMORY_MODE != "baseline":
            del q, k, qkv

    table = dequantize_epb_table(self, x.dtype)
    earth_position_bias = table[self.earth_position_index.view(-1)].view(
        window_tokens,
        window_tokens,
        self.num_pressure_height_windows,
        -1,
    )
    del table
    earth_position_bias = earth_position_bias.permute(3, 2, 0, 1).contiguous()
    softmax_applied = apply_cached_bias_mask_(attn, earth_position_bias, mask)
    del earth_position_bias

    if softmax_applied:
        pass
    elif use_inplace:
        torch.softmax(attn, dim=-1, out=attn)
    else:
        attn = self.softmax(attn)
    attn = self.attn_drop(attn)
    if deferred_v:
        v = F.linear(
            x,
            qkv_weight[2 * channels :],
            None if qkv_bias is None else qkv_bias[2 * channels :],
        )
        v = v.reshape(
            batch_width,
            num_pressure_height,
            window_tokens,
            self.num_heads,
            head_dim,
        ).permute(0, 3, 1, 2, 4)
        del qkv_weight, qkv_bias
    output = (attn @ v).permute(0, 2, 3, 1, 4).reshape(
        batch_width,
        num_pressure_height,
        window_tokens,
        channels,
    )
    output = self.proj(output)
    return self.proj_drop(output)


def reused_chunk_attention_forward(self, x: torch.Tensor, mask=None):
    """Reuse dequantized parameters and EPB across shallow width chunks."""
    batch_width, num_pressure_height, window_tokens, channels = x.shape
    chunk_width = ATTENTION_WIDTH_CHUNK
    head_dim = channels // self.num_heads
    output = torch.empty_like(x)
    qkv_weight, qkv_bias = dequantize_linear_parameters(self.qkv, x)
    proj_weight, proj_bias = dequantize_linear_parameters(self.proj, x)
    full_qkv_projection = CHUNK_PROJECTION_MODE == "full"
    full_output_projection = CHUNK_PROJECTION_MODE in {"output", "full"}
    qkv_all = None
    if full_qkv_projection:
        qkv_all = F.linear(x, qkv_weight, qkv_bias).reshape(
            batch_width,
            num_pressure_height,
            window_tokens,
            3,
            self.num_heads,
            head_dim,
        )
        if self.scale != 1.0:
            qkv_all[:, :, :, 0].mul_(self.scale)

    earth_position_bias = None
    earth_position_table = None
    if not REUSE_CHUNK_EPB_BIAS:
        earth_position_table = dequantize_epb_table(self, x.dtype)
    protected_widths = 0
    if (
        OUTPUT_BACKED_EPB_BIAS
        and earth_position_table is not None
        and CHUNK_PROJECTION_MODE == "chunk"
    ):
        bias_elements = (
            self.num_heads
            * num_pressure_height
            * window_tokens
            * window_tokens
        )
        elements_per_width = x[0].numel()
        protected_widths = (
            bias_elements + elements_per_width - 1
        ) // elements_per_width
        if protected_widths < batch_width:
            bias_storage = output.reshape(-1)[:bias_elements].view(
                self.num_heads,
                num_pressure_height,
                window_tokens * window_tokens,
            )
            table_by_head = earth_position_table.permute(2, 1, 0)
            gather_index = self.earth_position_index.reshape(
                1,
                1,
                -1,
            ).expand(
                self.num_heads,
                num_pressure_height,
                -1,
            )
            torch.gather(
                table_by_head,
                2,
                gather_index,
                out=bias_storage,
            )
            earth_position_bias = bias_storage.view(
                self.num_heads,
                num_pressure_height,
                window_tokens,
                window_tokens,
            )
            earth_position_table = None
        else:
            protected_widths = 0

    def get_position_bias() -> tuple[torch.Tensor | None, torch.Tensor | None]:
        nonlocal earth_position_bias
        if (
            REUSE_CHUNK_EPB_BIAS
            and earth_position_bias is None
        ):
            table = dequantize_epb_table(self, x.dtype)
            earth_position_bias = table[self.earth_position_index.view(-1)].view(
                window_tokens,
                window_tokens,
                self.num_pressure_height_windows,
                -1,
            )
            del table
            earth_position_bias = earth_position_bias.permute(
                3,
                2,
                0,
                1,
            ).contiguous()
        if earth_position_bias is not None:
            return earth_position_bias, None
        if earth_position_table is not None:
            chunk_bias = earth_position_table[
                self.earth_position_index.view(-1)
            ].view(
                window_tokens,
                window_tokens,
                self.num_pressure_height_windows,
                -1,
            )
            chunk_bias = chunk_bias.permute(3, 2, 0, 1).contiguous()
            return chunk_bias, chunk_bias
        return None, None

    def run_chunk(start: int, stop: int, chunk_mask=None) -> None:
        chunk_x = x[start:stop]
        chunk_batch_width = stop - start
        if full_qkv_projection:
            chunk_qkv = qkv_all[start:stop]
            q = chunk_qkv[:, :, :, 0].permute(0, 3, 1, 2, 4)
            k = chunk_qkv[:, :, :, 1].permute(0, 3, 1, 2, 4)
            v = chunk_qkv[:, :, :, 2].permute(0, 3, 1, 2, 4)
        else:
            qkv = F.linear(chunk_x, qkv_weight, qkv_bias).reshape(
                chunk_batch_width,
                num_pressure_height,
                window_tokens,
                3,
                self.num_heads,
                head_dim,
            ).permute(3, 0, 4, 1, 2, 5)
            q, k, v = qkv[0], qkv[1], qkv[2]
            apply_q_scale_(q, self.scale)

        if ATTENTION_KERNEL == "sdpa":
            v = v.contiguous()
            if not full_qkv_projection:
                del qkv
            position_bias, temporary_bias = get_position_bias()
            additive_mask = (
                None if position_bias is None else position_bias.unsqueeze(0)
            )
            if chunk_mask is not None:
                expanded_mask = chunk_mask.unsqueeze(1)
                additive_mask = (
                    expanded_mask
                    if additive_mask is None
                    else additive_mask + expanded_mask
                )
            chunk = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=additive_mask,
                dropout_p=0.0,
                scale=1.0,
            )
            del additive_mask, temporary_bias, position_bias, q, k
        else:
            attn = q @ k.transpose(-2, -1)
            v = v.contiguous()
            del q, k
            if not full_qkv_projection:
                del qkv
            position_bias, temporary_bias = get_position_bias()
            if position_bias is None:
                raise RuntimeError("attention EPB was not materialized")
            softmax_applied = apply_cached_bias_mask_(attn, position_bias, chunk_mask)
            del temporary_bias, position_bias
            if not softmax_applied:
                torch.softmax(attn, dim=-1, out=attn)
            attn = self.attn_drop(attn)
            chunk = attn @ v
            del attn
        del v
        chunk = chunk.permute(0, 2, 3, 1, 4).reshape(
            chunk_batch_width,
            num_pressure_height,
            window_tokens,
            channels,
        )
        if not full_output_projection:
            if CHUNK_LINEAR_OUT and not self.training:
                target = output[start:stop]
                input_2d = chunk.reshape(-1, channels)
                target_2d = target.reshape(-1, channels)
                if proj_bias is None:
                    torch.mm(
                        input_2d,
                        proj_weight.t(),
                        out=target_2d,
                    )
                else:
                    torch.addmm(
                        proj_bias,
                        input_2d,
                        proj_weight.t(),
                        out=target_2d,
                    )
                del target, input_2d, target_2d
            else:
                chunk = F.linear(chunk, proj_weight, proj_bias)
                chunk = self.proj_drop(chunk)
                output[start:stop].copy_(chunk)
        del chunk

    def finish() -> torch.Tensor:
        nonlocal qkv_all
        if not full_output_projection:
            return output
        qkv_all = None
        projected = F.linear(output, proj_weight, proj_bias)
        return self.proj_drop(projected)

    if mask is None:
        first_start = protected_widths if protected_widths else 0
        for start in range(first_start, batch_width, chunk_width):
            run_chunk(start, min(start + chunk_width, batch_width))
        if protected_widths:
            run_chunk(0, protected_widths)
        return finish()

    num_width_windows = mask.shape[0]
    if batch_width % num_width_windows:
        raise ValueError(
            f"attention batch_width={batch_width} is not divisible by "
            f"num_width_windows={num_width_windows}"
        )
    batches = batch_width // num_width_windows
    for batch_index in range(batches):
        batch_start = batch_index * num_width_windows
        width_begin = protected_widths if batch_index == 0 else 0
        for width_start in range(width_begin, num_width_windows, chunk_width):
            width_stop = min(width_start + chunk_width, num_width_windows)
            run_chunk(
                batch_start + width_start,
                batch_start + width_stop,
                mask[width_start:width_stop],
            )
    if protected_widths:
        run_chunk(0, protected_widths, mask[:protected_widths])
    return finish()


def quantized_epb_attention_forward(self, x: torch.Tensor, mask=None):
    """Optionally bound attention memory by processing width windows in chunks."""
    batch_width = x.shape[0]
    chunk_width = ATTENTION_WIDTH_CHUNK
    if (
        chunk_width <= 0
        or batch_width <= chunk_width
        or (x.shape[-1] > EMBED_DIM and not ATTENTION_CHUNK_DEEP)
        or (
            ATTENTION_CHUNK_STAGE != "all"
            and not any(
                self._attention_module_name.startswith(prefix)
                for prefix in ATTENTION_CHUNK_STAGE.split(",")
            )
        )
    ):
        return quantized_epb_attention_forward_full(self, x, mask)
    if REUSE_CHUNK_ATTENTION_PARAMETERS:
        return reused_chunk_attention_forward(self, x, mask)

    output = torch.empty_like(x)
    if mask is None:
        for start in range(0, batch_width, chunk_width):
            stop = min(start + chunk_width, batch_width)
            chunk = quantized_epb_attention_forward_full(self, x[start:stop], None)
            output[start:stop].copy_(chunk)
            del chunk
        return output

    num_width_windows = mask.shape[0]
    if batch_width % num_width_windows:
        raise ValueError(
            f"attention batch_width={batch_width} is not divisible by "
            f"num_width_windows={num_width_windows}"
        )
    batches = batch_width // num_width_windows
    for batch_index in range(batches):
        batch_start = batch_index * num_width_windows
        for width_start in range(0, num_width_windows, chunk_width):
            width_stop = min(width_start + chunk_width, num_width_windows)
            start = batch_start + width_start
            stop = batch_start + width_stop
            chunk = quantized_epb_attention_forward_full(
                self,
                x[start:stop],
                mask[width_start:width_stop],
            )
            output[start:stop].copy_(chunk)
            del chunk
    return output


def replace_modules_with_quantized_epb(
    model: nn.Module,
    state: dict[str, torch.Tensor],
) -> int:
    """Replace fp16 EPB parameters with checkpoint-backed compressed buffers."""
    replaced = 0
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        int8_key = f"{prefix}earth_position_bias_qweight"
        int4_key = f"{prefix}earth_position_bias_qweight_packed"
        scale_key = f"{prefix}earth_position_bias_scale"
        if int8_key not in state and int4_key not in state:
            continue
        if scale_key not in state:
            raise KeyError(f"missing EPB scale: {scale_key}")
        if not hasattr(module, "earth_position_bias_table"):
            raise AttributeError(f"{module_name} has no earth_position_bias_table")

        original_shape = tuple(module.earth_position_bias_table.shape)
        rows = original_shape[0]
        del module._parameters["earth_position_bias_table"]
        if int8_key in state:
            module.register_buffer(
                "earth_position_bias_qweight",
                torch.empty_like(state[int8_key]),
            )
        else:
            module.register_buffer(
                "earth_position_bias_qweight_packed",
                torch.empty_like(state[int4_key]),
            )
        num_blocks = state[scale_key].shape[0]
        if rows % num_blocks != 0:
            raise ValueError(
                f"invalid EPB blocks for {module_name}: "
                f"rows={rows}, blocks={num_blocks}"
            )
        module.register_buffer(
            "earth_position_bias_scale",
            torch.empty_like(state[scale_key]),
        )
        module._epb_block_size = rows // num_blocks
        module._epb_original_shape = original_shape
        module._attention_module_name = module_name
        module.forward = MethodType(quantized_epb_attention_forward, module)
        replaced += 1
    return replaced


def unpack_int4_epb_once(model: nn.Module) -> int:
    """Expand packed INT4 EPB to INT8 once, avoiding unpack work per sample.

    The checkpoint remains packed on disk. At runtime the packed buffer is
    replaced, so the attention path gets INT8-like speed without retaining both
    representations.
    """
    unpacked = 0
    for module in model.modules():
        if "earth_position_bias_qweight_packed" not in module._buffers:
            continue
        packed = module.earth_position_bias_qweight_packed
        rows, pressure_height, heads = module._epb_original_shape
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        quantized = (
            torch.stack((low, high), dim=1)
            .reshape(rows, pressure_height, heads)
            .to(torch.int8)
            .sub_(8)
            .contiguous()
        )
        del module._buffers["earth_position_bias_qweight_packed"]
        module.register_buffer(
            "earth_position_bias_qweight",
            quantized,
            persistent=False,
        )
        unpacked += 1
    return unpacked


def build_model(cfg, img_size: list[int], device: torch.device) -> Pangu:
    model = Pangu(
        img_size=img_size,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        window_size=cfg.window_size,
    )
    model.patchrecovery2d = OneRecovery(
        style="PanguPatchRecovery",
        img_size=img_size,
        patch_size=PATCH_SIZE[1:],
        in_chans=EMBED_DIM * 2,
        out_chans=4,
    )
    model.patchrecovery3d = OneRecovery(
        style="PanguPatchRecovery",
        img_size=(13, *img_size),
        patch_size=PATCH_SIZE,
        in_chans=EMBED_DIM * 2,
        out_chans=5,
    )
    return model.to(device=device, dtype=DTYPE)


def load_model(cfg, checkpoint_path: str | Path, img_size: list[int], device: torch.device) -> Pangu:
    model = build_model(cfg, img_size, device)
    state = load_checkpoint_state(checkpoint_path)
    if is_w8a16_state(state):
        replace_modules_with_w8a16(model)
        print("checkpoint format: W8A16")
        unpacked_w4 = unpack_w4_linear_state(model, state)
        if unpacked_w4:
            print(
                f"checkpoint low-bit storage: blockwise "
                f"({unpacked_w4} Linear modules)"
            )
    else:
        print("warning: checkpoint does not look like W8A16; running with fp weights")
    if is_quantized_epb_state(state):
        replaced_epb = replace_modules_with_quantized_epb(model, state)
        print(f"checkpoint EPB format: quantized ({replaced_epb} attention modules)")
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = (
        "attn_mask",
        "earth_position_index",
        "device_buffer",
        ".Fuser.",
        ".Sampler.",
        ".Reconvery.",
    )
    bad_missing = [key for key in missing if not any(x in key for x in allowed_missing)]
    if bad_missing or unexpected:
        raise RuntimeError(
            "Unexpected checkpoint mismatch:\n"
            f"bad_missing={bad_missing[:20]}\n"
            f"unexpected={unexpected[:20]}"
        )
    model = model.to(device=device, dtype=DTYPE)
    optimized_masks, original_mask_bytes, optimized_mask_bytes = (
        optimize_attention_masks(model)
    )
    if optimized_masks:
        print(
            f"runtime attention masks: {ATTENTION_MASK_MODE} "
            f"({optimized_masks} buffers, "
            f"{original_mask_bytes / 1e6:.2f} -> "
            f"{optimized_mask_bytes / 1e6:.2f} MB unique)"
        )
    if ATTENTION_BIASMASK_MODE.startswith("fused_"):
        load_fused_bias_mask_add()
        print(
            "runtime attention bias+mask: "
            f"{ATTENTION_BIASMASK_MODE} FP16+INT8 kernel"
        )
    unpacked_epb = unpack_int4_epb_once(model)
    if unpacked_epb:
        print(f"runtime EPB cache: unpacked INT4 to INT8 ({unpacked_epb} modules)")
    cached_epb, epb_cache_added_bytes = cache_epb_tables(model, EPB_CACHE_MODE)
    if cached_epb:
        print(
            f"runtime EPB table cache: {EPB_CACHE_MODE} "
            f"({cached_epb} modules, net {epb_cache_added_bytes / 1e6:.2f} MB)"
        )
    cached_linears, cache_added_bytes = cache_linear_weights(model, LINEAR_CACHE_MODE)
    if cached_linears:
        print(
            f"runtime Linear cache: {LINEAR_CACHE_MODE} "
            f"({cached_linears} modules, net {cache_added_bytes / 1e6:.2f} MB)"
        )
    folded_norms = fold_layernorm_affine(model)
    if folded_norms:
        print(f"runtime normalization affine folding: {folded_norms} modules")
    gfx936_norms = install_gfx936_layer_norm(model)
    if gfx936_norms:
        print(f"runtime LayerNorm: gfx936 fixed-width kernel ({gfx936_norms} modules)")
    folded_q_scales = fold_attention_q_scale(model)
    if folded_q_scales:
        print(f"runtime attention Q-scale folding: {folded_q_scales} modules")
    cached_convs, conv_cache_added_bytes = cache_convolution_weights(model)
    if cached_convs:
        print(
            f"runtime convolution cache: "
            f"{cached_convs} modules, net {conv_cache_added_bytes / 1e6:.2f} MB"
        )
    indexed_roll_blocks = install_roll_indices(model, device)
    if indexed_roll_blocks:
        print(f"runtime cyclic shift: indexed ({indexed_roll_blocks} blocks)")
    fused_window_blocks = install_fused_window_indices(model, device)
    if fused_window_blocks:
        print(f"runtime shifted windows: fused ({fused_window_blocks} blocks)")
    direct_reverse_blocks = install_direct_reverse_indices(model, device)
    if direct_reverse_blocks:
        print(
            f"runtime window reverse+crop: indexed "
            f"({direct_reverse_blocks} blocks)"
        )
    if WINDOW_GATHER_MODE == "gfx936" and direct_reverse_blocks:
        load_fused_bias_mask_add()
        print("runtime window gather: gfx936 FP16 token kernel")
    if LEAN_BLOCK_FORWARD:
        lean_blocks = 0
        for module in model.modules():
            if module.__class__.__name__ != "EarthTransformer3DBlock":
                continue
            module.forward = MethodType(lean_earth_transformer_forward, module)
            module.mlp.forward = MethodType(lean_mlp_forward, module.mlp)
            lean_blocks += 1
        print(f"runtime transformer path: lean buffers ({lean_blocks} blocks)")
    if OFFLOAD_SKIP_TOKENS and device.type == "cuda":
        model._skip_copy_stream = torch.cuda.Stream(device=device)
        print("runtime skip path: pinned-CPU asynchronous offload")
    model.forward = MethodType(lean_pangu_forward, model)
    if RECOVERY3D_MODE == "chunked":
        model.patchrecovery3d = ChunkedPatchRecovery3D(
            model.patchrecovery3d,
            chunk_width=RECOVERY3D_CHUNK_WIDTH,
        )
    model.eval()
    return model


def build_test_dataloader(cfg_data):
    try:
        datapipe = ERA5Datapipe(params=cfg_data, distributed=False)
        return datapipe.test_dataloader()
    except (TypeError, AttributeError):
        datapipe = ERA5Datapipe(
            dataset_dir=cfg_data.dataset.data_dir,
            used_variables=cfg_data.dataset.channels,
            used_years=cfg_data.dataset.test_time,
            distributed=False,
            batch_size=cfg_data.dataloader.batch_size,
            num_workers=cfg_data.dataloader.num_workers,
        )
        test_dataloader, _ = datapipe.get_dataloader("test")
        return test_dataloader


def build_surface_mask(cfg_data, device: torch.device) -> torch.Tensor:
    static_dir = cfg_data.dataset.static_dir
    land_mask = torch.from_numpy(
        np.load(os.path.join(static_dir, "land_mask.npy")).astype(np.float32)
    )
    soil_type = torch.from_numpy(
        np.load(os.path.join(static_dir, "soil_type.npy")).astype(np.float32)
    )
    topography = torch.from_numpy(
        np.load(os.path.join(static_dir, "topography.npy")).astype(np.float32)
    )
    topography = (topography - topography.mean()) / (
        topography.std(unbiased=False) + 1e-6
    )
    return torch.stack([land_mask, soil_type, topography], dim=0).unsqueeze(0).to(
        device=device,
        dtype=DTYPE,
    )


def prepare_model_input(
    invar: torch.Tensor,
    surface_mask: torch.Tensor,
    model_img_size: Iterable[int],
    device: torch.device,
    destination: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Move and assemble all 69 input channels for model inference."""
    if INPUT_ASSEMBLY_MODE == "direct":
        batch, channels, height, width = invar.shape
        target_height, target_width = list(model_img_size)
        left, right, top, bottom = symmetric_padding(
            (height, width),
            (target_height, target_width),
        )
        expected_shape = (batch, channels + 3, target_height, target_width)
        if destination is None:
            model_input = torch.empty(
                expected_shape,
                device=device,
                dtype=DTYPE,
            )
        else:
            if tuple(destination.shape) != expected_shape:
                raise ValueError(
                    "input destination shape mismatch: "
                    f"expected {expected_shape}, got {tuple(destination.shape)}"
                )
            if destination.device != device or destination.dtype != DTYPE:
                raise ValueError("input destination device/dtype mismatch")
            model_input = destination
        center = model_input[
            :,
            :,
            top : top + height,
            left : left + width,
        ]
        center[:, :4].copy_(invar[:, :4], non_blocking=True)
        center[:, 4:7].copy_(
            surface_mask.expand(batch, -1, -1, -1),
        )
        center[:, 7:].copy_(invar[:, 4:], non_blocking=True)
        if left:
            model_input[:, :, top : top + height, :left].copy_(
                center[:, :, :, :1].expand(-1, -1, -1, left)
            )
        if right:
            model_input[:, :, top : top + height, left + width :].copy_(
                center[:, :, :, -1:].expand(-1, -1, -1, right)
            )
        if top:
            model_input[:, :, :top].copy_(
                model_input[:, :, top : top + 1].expand(-1, -1, top, -1)
            )
        if bottom:
            model_input[:, :, top + height :].copy_(
                model_input[:, :, top + height - 1 : top + height].expand(
                    -1,
                    -1,
                    bottom,
                    -1,
                )
            )
        return model_input, channels - 4

    invar_surface = invar[:, :4].to(device=device, dtype=DTYPE)
    invar_upper_air = invar[:, 4:].to(device=device, dtype=DTYPE)
    upper_channels = invar_upper_air.shape[1]
    batch_mask = surface_mask.expand(invar_surface.shape[0], -1, -1, -1)
    model_input = torch.cat(
        [invar_surface, batch_mask, invar_upper_air],
        dim=1,
    )
    del invar_surface, invar_upper_air, batch_mask
    padded = pad_to_img_size(model_input, model_img_size)
    if destination is None:
        return padded, upper_channels
    if tuple(destination.shape) != tuple(padded.shape):
        raise ValueError("input destination shape mismatch")
    destination.copy_(padded)
    return destination, upper_channels


def _replicate_pad_input_(
    tensor: torch.Tensor,
    *,
    height: int,
    width: int,
    left: int,
    right: int,
    top: int,
    bottom: int,
) -> None:
    center = tensor[:, :, top : top + height, left : left + width]
    if left:
        tensor[:, :, top : top + height, :left].copy_(
            center[:, :, :, :1].expand(-1, -1, -1, left)
        )
    if right:
        tensor[:, :, top : top + height, left + width :].copy_(
            center[:, :, :, -1:].expand(-1, -1, -1, right)
        )
    if top:
        tensor[:, :, :top].copy_(
            tensor[:, :, top : top + 1].expand(-1, -1, top, -1)
        )
    if bottom:
        tensor[:, :, top + height :].copy_(
            tensor[:, :, top + height - 1 : top + height].expand(
                -1, -1, bottom, -1
            )
        )


def prepare_split_model_input(
    invar: torch.Tensor,
    surface_mask: torch.Tensor,
    model_img_size: Iterable[int],
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], int]:
    """Build the two patch-embedding inputs without a 72-channel staging copy."""
    batch, channels, height, width = invar.shape
    target_height, target_width = list(model_img_size)
    left, right, top, bottom = symmetric_padding(
        (height, width),
        (target_height, target_width),
    )
    upper_channels = channels - 4
    surface_input = torch.empty(
        (batch, 7, target_height, target_width),
        device=device,
        dtype=DTYPE,
    )
    upper_input = torch.empty(
        (batch, upper_channels, target_height, target_width),
        device=device,
        dtype=DTYPE,
    )
    surface_center = surface_input[
        :, :, top : top + height, left : left + width
    ]
    upper_center = upper_input[:, :, top : top + height, left : left + width]
    surface_center[:, :4].copy_(invar[:, :4], non_blocking=True)
    surface_center[:, 4:].copy_(surface_mask.expand(batch, -1, -1, -1))
    upper_center.copy_(invar[:, 4:], non_blocking=True)
    _replicate_pad_input_(
        surface_input,
        height=height,
        width=width,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
    )
    _replicate_pad_input_(
        upper_input,
        height=height,
        width=width,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
    )
    upper_input = upper_input.reshape(
        batch,
        5,
        upper_channels // 5,
        target_height,
        target_width,
    )
    return (surface_input, upper_input), upper_channels


def prepare_streamed_surface_input(
    invar: torch.Tensor,
    surface_mask: torch.Tensor,
    model_img_size: Iterable[int],
    device: torch.device,
) -> torch.Tensor:
    batch, _channels, height, width = invar.shape
    target_height, target_width = list(model_img_size)
    left, right, top, bottom = symmetric_padding(
        (height, width),
        (target_height, target_width),
    )
    surface = torch.empty(
        (batch, 7, target_height, target_width),
        device=device,
        dtype=DTYPE,
    )
    center = surface[:, :, top : top + height, left : left + width]
    center[:, :4].copy_(invar[:, :4], non_blocking=True)
    center[:, 4:].copy_(surface_mask.expand(batch, -1, -1, -1))
    _replicate_pad_input_(
        surface,
        height=height,
        width=width,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
    )
    return surface


def prepare_streamed_upper_input(
    invar: torch.Tensor,
    model_img_size: Iterable[int],
    device: torch.device,
) -> torch.Tensor:
    batch, channels, height, width = invar.shape
    target_height, target_width = list(model_img_size)
    left, right, top, bottom = symmetric_padding(
        (height, width),
        (target_height, target_width),
    )
    upper_channels = channels - 4
    upper = torch.empty(
        (batch, upper_channels, target_height, target_width),
        device=device,
        dtype=DTYPE,
    )
    upper[:, :, top : top + height, left : left + width].copy_(
        invar[:, 4:],
        non_blocking=True,
    )
    _replicate_pad_input_(
        upper,
        height=height,
        width=width,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
    )
    return upper.reshape(
        batch,
        5,
        upper_channels // 5,
        target_height,
        target_width,
    )


def denormalize_prediction(
    prediction: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    """Denormalize all 69 outputs without allocating an extra expression result."""
    prediction = prediction.astype(np.float32, copy=False)
    np.multiply(prediction, stds, out=prediction)
    np.add(prediction, means, out=prediction)
    return prediction


def initialize_output_pipeline(device: torch.device) -> None:
    global _OUTPUT_COPY_STREAM, _OUTPUT_READY_EVENTS, _OUTPUT_COPY_DONE_EVENTS
    if not OUTPUT_PIPELINE:
        return
    if device.type != "cuda":
        raise ValueError("output pipeline requires a CUDA/HIP device")
    _OUTPUT_COPY_STREAM = torch.cuda.Stream(device=device)
    _OUTPUT_READY_EVENTS = [
        torch.cuda.Event(blocking=False) for _ in range(OUTPUT_PIPELINE_BUFFERS)
    ]
    _OUTPUT_COPY_DONE_EVENTS = [
        torch.cuda.Event(blocking=False) for _ in range(OUTPUT_PIPELINE_BUFFERS)
    ]


def transfer_prediction_to_cpu_pipelined(
    prediction: torch.Tensor,
    output_buffer: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
) -> np.ndarray:
    if (
        _OUTPUT_COPY_STREAM is None
        or _OUTPUT_READY_EVENTS is None
        or _OUTPUT_COPY_DONE_EVENTS is None
    ):
        raise RuntimeError("output pipeline was not initialized")

    target = output_buffer[: prediction.shape[0]]
    scratch = [
        torch.empty(
            prediction.shape[0],
            OUTPUT_TRANSFER_CHUNK,
            prediction.shape[2],
            prediction.shape[3],
            device=prediction.device,
            dtype=torch.float32,
        )
        for _ in range(OUTPUT_PIPELINE_BUFFERS)
    ]
    current_stream = torch.cuda.current_stream(prediction.device)
    for chunk_index, start in enumerate(
        range(0, prediction.shape[1], OUTPUT_TRANSFER_CHUNK)
    ):
        stop = min(start + OUTPUT_TRANSFER_CHUNK, prediction.shape[1])
        width = stop - start
        buffer_index = chunk_index % OUTPUT_PIPELINE_BUFFERS
        if chunk_index >= OUTPUT_PIPELINE_BUFFERS:
            current_stream.wait_event(_OUTPUT_COPY_DONE_EVENTS[buffer_index])

        source = scratch[buffer_index][:, :width]
        chunk = prediction[:, start:stop]
        if OUTPUT_AFFINE_MODE == "gfx936":
            source = gfx936_affine_fp16_to_fp32(
                chunk,
                means[:, start:stop],
                stds[:, start:stop],
                source,
            )
        else:
            source.copy_(chunk)
            source.mul_(stds[:, start:stop])
            source.add_(means[:, start:stop])
        _OUTPUT_READY_EVENTS[buffer_index].record(current_stream)
        with torch.cuda.stream(_OUTPUT_COPY_STREAM):
            _OUTPUT_COPY_STREAM.wait_event(_OUTPUT_READY_EVENTS[buffer_index])
            target[:, start:stop].copy_(source, non_blocking=True)
            source.record_stream(_OUTPUT_COPY_STREAM)
            _OUTPUT_COPY_DONE_EVENTS[buffer_index].record(_OUTPUT_COPY_STREAM)
    current_stream.wait_stream(_OUTPUT_COPY_STREAM)
    return target.numpy()


def transfer_prediction_to_cpu(
    prediction: torch.Tensor,
    output_buffer: torch.Tensor,
    means: torch.Tensor | None = None,
    stds: torch.Tensor | None = None,
) -> np.ndarray:
    """Convert and copy outputs in bounded chunks to pinned FP32 CPU memory.

    Reuses one device FP32 scratch so denormalize temporaries do not stack in
    the caching allocator across chunks.
    """
    if (
        OUTPUT_PIPELINE
        and prediction.is_cuda
        and means is not None
        and stds is not None
    ):
        return transfer_prediction_to_cpu_pipelined(
            prediction,
            output_buffer,
            means,
            stds,
        )
    target = output_buffer[: prediction.shape[0]]
    scratch: torch.Tensor | None = None
    for start in range(0, prediction.shape[1], OUTPUT_TRANSFER_CHUNK):
        stop = min(start + OUTPUT_TRANSFER_CHUNK, prediction.shape[1])
        width = stop - start
        chunk = prediction[:, start:stop]
        if means is not None and stds is not None:
            if scratch is None or scratch.shape[1] < width:
                scratch = torch.empty(
                    prediction.shape[0],
                    width,
                    prediction.shape[2],
                    prediction.shape[3],
                    device=prediction.device,
                    dtype=torch.float32,
                )
            source = scratch[:, :width]
            if OUTPUT_AFFINE_MODE == "gfx936":
                source = gfx936_affine_fp16_to_fp32(
                    chunk,
                    means[:, start:stop],
                    stds[:, start:stop],
                    source,
                )
            else:
                source.copy_(chunk)
                source.mul_(stds[:, start:stop])
                source.add_(means[:, start:stop])
        else:
            source = chunk
        target[:, start:stop].copy_(source, non_blocking=True)
        if prediction.is_cuda:
            torch.cuda.synchronize(prediction.device)
    return target.numpy()


class CUDAGraphModelRunner:
    """Capture fixed-shape model execution using sample-independent zeros."""

    @torch.inference_mode()
    def __init__(
        self,
        model: nn.Module,
        input_shape: tuple[int, int, int, int],
        device: torch.device,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("cuda_graph execution requires a CUDA/HIP device")
        self.model = model
        self.static_input = torch.zeros(
            input_shape,
            device=device,
            dtype=DTYPE,
        )

        current_stream = torch.cuda.current_stream(device)
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(2):
                warm_surface, warm_upper = model(self.static_input)
                del warm_surface, warm_upper
        current_stream.wait_stream(warmup_stream)
        torch.cuda.synchronize(device)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output_surface, self.output_upper = model(self.static_input)
        torch.cuda.synchronize(device)

    def replay(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.graph.replay()
        return self.output_surface, self.output_upper


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    current_path = os.getcwd()
    sys.path.append(current_path)

    config_file_path = os.path.join(current_path, args.config)
    cfg = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")
    if PIN_INPUT_MEMORY:
        cfg_data.dataloader.pin_memory = True

    device = torch.device(DEVICE)
    output_img_size = list(cfg_data.dataset.img_size)
    model_img_size = padded_img_size(output_img_size, PATCH_SIZE)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, cfg)
    output_dir = Path(args.output_dir)
    time_record_path = Path(args.time_record)
    memory_record_path = "" if args.disable_torch_memory else args.memory_record

    means, stds = get_stats(cfg_data.dataset.data_dir, cfg_data.dataset.channels)
    if OUTPUT_DENORMALIZE_MODE == "gpu":
        means_device = torch.from_numpy(means).to(device=device, dtype=torch.float32)
        stds_device = torch.from_numpy(stds).to(device=device, dtype=torch.float32)
    else:
        means_device = None
        stds_device = None
    test_dataloader = build_test_dataloader(cfg_data)
    surface_mask = build_surface_mask(cfg_data, device)
    model = load_model(cfg, checkpoint_path, model_img_size, device)
    initialize_output_pipeline(device)
    graph_runner = None
    if (SPLIT_MODEL_INPUT or STREAMED_MODEL_INPUT) and EXECUTION_MODE == "cuda_graph":
        raise ValueError("split/streamed model input is incompatible with cuda_graph mode")
    if EXECUTION_MODE == "cuda_graph":
        graph_runner = CUDAGraphModelRunner(
            model,
            (
                cfg_data.dataloader.batch_size,
                len(cfg_data.dataset.channels) + 3,
                model_img_size[0],
                model_img_size[1],
            ),
            device,
        )
    output_cpu_buffer = torch.empty(
        cfg_data.dataloader.batch_size,
        len(cfg_data.dataset.channels),
        output_img_size[0],
        output_img_size[1],
        dtype=torch.float32,
        device="cpu",
        pin_memory=device.type == "cuda",
    )
    if EMPTY_CACHE_BEFORE_INFERENCE and device.type == "cuda":
        torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    time_record_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"samples will be generated to './{output_dir.as_posix()}/'")
    print(
        "Patch-Lite submit: "
        f"patch_size={PATCH_SIZE}, embed_dim={EMBED_DIM}, num_heads={NUM_HEADS}, "
        f"model_img_size={model_img_size}, dtype=fp16, recovery3d=all69-{RECOVERY3D_MODE}, "
        f"chunk_width={RECOVERY3D_CHUNK_WIDTH}, warmup_steps={WARMUP_STEPS}, "
        f"attention_memory={ATTENTION_MEMORY_MODE}, "
        f"attention_width_chunk={ATTENTION_WIDTH_CHUNK}, "
        f"attention_chunk_deep={int(ATTENTION_CHUNK_DEEP)}, "
        f"attention_chunk_stage={ATTENTION_CHUNK_STAGE}, "
        f"mlp_token_chunk={MLP_TOKEN_CHUNK}, "
        f"mlp_chunk_deep={int(MLP_CHUNK_DEEP)}, "
        f"offload_skip_tokens={int(OFFLOAD_SKIP_TOKENS)}, "
        f"linear_cache={LINEAR_CACHE_MODE}, "
        f"reuse_chunk_parameters={int(REUSE_CHUNK_ATTENTION_PARAMETERS)}, "
        f"reuse_chunk_epb_bias={int(REUSE_CHUNK_EPB_BIAS)}, "
        f"output_backed_epb_bias={int(OUTPUT_BACKED_EPB_BIAS)}, "
        f"epb_cache={EPB_CACHE_MODE}, "
        f"roll_mode={ROLL_MODE}, "
        f"chunk_projection={CHUNK_PROJECTION_MODE}, "
        f"chunk_linear_out={int(CHUNK_LINEAR_OUT)}, "
        f"shift_window={SHIFT_WINDOW_MODE}, "
        f"direct_window_forward={int(DIRECT_WINDOW_FORWARD)}, "
        f"direct_window_reverse={int(DIRECT_WINDOW_REVERSE)}, "
        f"attention_kernel={ATTENTION_KERNEL}, "
        f"attention_biasmask={ATTENTION_BIASMASK_MODE}, "
        f"attention_mask={ATTENTION_MASK_MODE}, "
        f"layer_norm={LAYER_NORM_MODE}, "
        f"window_gather={WINDOW_GATHER_MODE}, "
        f"fuse_norm_gather={int(FUSE_LAYER_NORM_GATHER)}, "
        f"conv_cache={int(CONV_CACHE)}, "
        f"fold_norm_affine={int(FOLD_NORM_AFFINE)}, "
        f"fold_resample_norm_affine={int(FOLD_RESAMPLE_NORM_AFFINE)}, "
        f"fold_q_scale={int(FOLD_Q_SCALE)}, "
        f"gelu_approximate={GELU_APPROXIMATE}, "
        f"mlp_residual_out={int(MLP_RESIDUAL_OUT)}, "
        f"input_assembly={INPUT_ASSEMBLY_MODE}, "
        f"split_model_input={int(SPLIT_MODEL_INPUT)}, "
        f"streamed_model_input={int(STREAMED_MODEL_INPUT)}, "
        f"output_transfer_chunk={OUTPUT_TRANSFER_CHUNK}, "
        f"output_pipeline={int(OUTPUT_PIPELINE)}, "
        f"output_assembly={OUTPUT_ASSEMBLY_MODE}, "
        f"output_affine={OUTPUT_AFFINE_MODE}, "
        f"output_denormalize={OUTPUT_DENORMALIZE_MODE}, "
        f"pin_input_memory={int(PIN_INPUT_MEMORY)}, "
        f"empty_cache_before_inference={int(EMPTY_CACHE_BEFORE_INFERENCE)}, "
        f"execution_mode={EXECUTION_MODE}, "
        f"lean_forward=1, lean_blocks={int(LEAN_BLOCK_FORWARD)}, "
        f"timing_scope=post-loader-to-prediction"
    )

    time_list = []
    memory_samples = []
    first = True
    num_test_batches = len(test_dataloader)
    test_years = cfg_data.dataset.test_ratio
    skip_sample_names = get_skip_sample_names(
        data_dir=cfg_data.dataset.data_dir,
        test_years=test_years,
        skip_count=FORECAST_STEPS-1,
    )
    num_inference_batches = max(
        num_test_batches - len(skip_sample_names), 0
    )
    print(
        f"Inferring {num_inference_batches} samples; "
        f"skipping {len(skip_sample_names)} samples "
        f"from {len(test_years)} years"
    )
    progress = tqdm(
        total=num_inference_batches,
        desc="Inferring testset",
        unit="sample",
    )
    skipped_seen = set()
    processed_samples = 0

    with torch.inference_mode():
        for data in test_dataloader:
            invar = data[0]
            filename = data[4][0][0]
            if filename in skip_sample_names:
                skipped_seen.add(filename)
                continue
            
            if first:
                first = False
                print(f"  invar  shape: {list(invar.shape)}   -> [batch, channels, H, W]")
                print(f"  the first filename: {filename}")

            torch.cuda.synchronize()
            if memory_record_path:
                torch.cuda.reset_peak_memory_stats(device)
            # ----------------------AI4S(时间度量不可更改)---------------------------
            start_time = time.perf_counter()
            output_dir=f"result/output/{filename}"
            if invar.ndim != 4 or invar.shape[0] != 1:
                raise ValueError(
                    f"推理要求 batch_size=1，实际输入形状为 {tuple(initial_state.shape)}"
                )
            batch_size, num_channels, _, _ = invar.shape
            if num_channels != 69:
                raise ValueError(f"期望 69 个气象变量，实际得到 {num_channels} 个")
            
            # 每个初始时刻使用一个独立目录，目录内保存 8 个预报时效。
            os.makedirs(output_dir, exist_ok=True)
        
            state = invar.to("cuda:0", dtype=torch.float32)
            static = surface_mask[:batch_size]

            del invar

            for step in range(FORECAST_STEPS):
                state=crop_to_img_size(state, output_img_size)
                print(state.shape)
                if STREAMED_MODEL_INPUT:
                    upper_channels = state.shape[1] - 4
                    model_input = (state, surface_mask, tuple(model_img_size))
                elif SPLIT_MODEL_INPUT:
                    model_input, upper_channels = prepare_split_model_input(
                        state,
                        surface_mask,
                        model_img_size,
                        device,
                    )
                else:
                    model_input, upper_channels = prepare_model_input(
                        state,
                        surface_mask,
                        model_img_size,
                        device,
                        destination=(
                            None if graph_runner is None else graph_runner.static_input
                        ),
                    )
                if graph_runner is None:
                    # Drop the caller ref before forward so lean_forward can free input storage.
                    _input_box = [model_input]
                    del model_input
                    out_surface, out_upper_air = model(_input_box.pop())
                    del _input_box
                else:
                    del model_input
                    out_surface, out_upper_air = graph_runner.replay()
                out_upper_air = out_upper_air.reshape(
                    out_surface.shape[0],
                    upper_channels,
                    model_img_size[0],
                    model_img_size[1],
                )
                print(out_surface.shape, out_upper_air.shape)
                state = torch.cat([out_surface, out_upper_air], dim=1)
                if OUTPUT_ASSEMBLY_MODE == "split":
                    batch_size = out_surface.shape[0]
                    surface_channels = out_surface.shape[1]
                    transfer_prediction_to_cpu(
                        crop_to_img_size(out_surface, output_img_size),
                        output_cpu_buffer[:, :surface_channels],
                        (
                            None
                            if means_device is None
                            else means_device[:, :surface_channels]
                        ),
                        (
                            None
                            if stds_device is None
                            else stds_device[:, :surface_channels]
                        ),
                    )
                    del out_surface
                    transfer_prediction_to_cpu(
                        crop_to_img_size(out_upper_air, output_img_size),
                        output_cpu_buffer[:, surface_channels:],
                        (
                            None
                            if means_device is None
                            else means_device[:, surface_channels:]
                        ),
                        (
                            None
                            if stds_device is None
                            else stds_device[:, surface_channels:]
                        ),
                    )
                    del out_upper_air
                    pred_var = output_cpu_buffer[:batch_size].numpy()
                else:
                    pred_var = torch.cat([out_surface, out_upper_air], dim=1)
                    del out_surface, out_upper_air
                    pred_var = transfer_prediction_to_cpu(
                        crop_to_img_size(pred_var, output_img_size),
                        output_cpu_buffer,
                        means_device,
                        stds_device,
                    )
                if OUTPUT_DENORMALIZE_MODE == "cpu":
                    pred_var = denormalize_prediction(pred_var, means, stds)
                lead_hour = int((step + 1) * FORECAST_INTERVAL_HOURS)
                np.save(os.path.join(output_dir, f"{lead_hour:03d}h.npy"), pred_var)
                # 释放当前时效的CPU内存
                del pred_var
            torch.cuda.synchronize()
            if "state" in locals():
                del state
            gc.collect()
            torch.cuda.empty_cache()
            end_time = time.perf_counter()
            time_list.append(end_time - start_time)
            processed_samples += 1
            progress.update(1)
            if memory_record_path:
                memory_samples.append(
                    {
                        "allocated_bytes": torch.cuda.memory_allocated(device),
                        "reserved_bytes": torch.cuda.memory_reserved(device),
                        "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    }
                )
            # ---------------------------------------------------------------------
        progress.close()
        missing_skip_names = skip_sample_names - skipped_seen
        if missing_skip_names:
            raise RuntimeError(
                "以下计划跳过的样本未出现在 dataloader 中: "
                f"{sorted(missing_skip_names)}"
            )
        if processed_samples != num_inference_batches:
            raise RuntimeError(
                f"实际推理样本数 {processed_samples} 与预期 "
                f"{num_inference_batches} 不一致"
            )

        with open(time_record_path, "w", encoding="utf-8") as f:
            json.dump(time_list, f, ensure_ascii=False, indent=4)
        if memory_record_path:
            memory_path = Path(memory_record_path)
            memory_path.parent.mkdir(parents=True, exist_ok=True)
            with open(memory_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "summary": summarize_memory_samples(memory_samples),
                        "samples": memory_samples,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )


if __name__ == "__main__":
    main()

