#!/usr/bin/env python3
"""V2 Patch-Lite distillation training — decomposed loss, strict init, metadata & resume.

Decomposed loss:
    total = w_all_teacher * loss_teacher_all
          + w_15_teacher  * loss_teacher_15
          + w_15_gt       * loss_gt_15
          + w_all_gt      * loss_gt_all
          + alpha_feature * loss_feature

Loss presets:
    teacher15              — focus on official-15 teacher distillation (default)
    teacher15_gt005        — teacher15 + small GT on official-15
    teacher15_gt010        — teacher15 + larger GT on official-15
    legacy_weighted_all69  — exact old behaviour for backward compatibility

Init modes:
    random                     — random init (smoke test only)
    shape_match                — inherit same-name same-shape tensors
    teacher_slice              — structural embed-dim slicing from teacher
    teacher_slice_interpolate  — teacher_slice + spatial interpolation for patch kernels
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import random as pyrandom
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from onescience.datapipes.climate.era5_old import ERA5Datapipe
from onescience.models.pangu import Pangu
from onescience.modules import OneRecovery
from onescience.utils.YParams import YParams
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
DEFAULT_OFFICIAL_15 = [0, 1, 2, 3, 6, 7, 9, 19, 20, 22, 32, 33, 35, 48, 61]
STAGE_ORDER = ("layer1", "layer2", "layer3", "layer4")
DEFAULT_STAGE_WEIGHTS = {"layer1": 0.15, "layer2": 0.5, "layer3": 0.5, "layer4": 0.15}
DROP_PARAM_ONLY_PATTERNS = (
    ".attn_mask", ".earth_position_index", "device_buffer",
    ".Fuser.", ".Sampler.", ".Reconvery.",
)
INIT_MODES = ("random", "shape_match", "teacher_slice", "teacher_slice_interpolate")

LOSS_PRESETS: dict[str, dict[str, float]] = {
    "teacher15":            {"w_all_teacher": 0.2, "w_15_teacher": 1.0, "w_15_gt": 0.0,  "w_all_gt": 0.0},
    "teacher15_gt005":      {"w_all_teacher": 0.2, "w_15_teacher": 1.0, "w_15_gt": 0.05, "w_all_gt": 0.0},
    "teacher15_gt010":      {"w_all_teacher": 0.2, "w_15_teacher": 1.0, "w_15_gt": 0.10, "w_all_gt": 0.0},
    "legacy_weighted_all69": {"w_all_teacher": 1.0, "w_15_teacher": 0.0, "w_15_gt": 0.0,  "w_all_gt": 0.05},
}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def script_sha256() -> str:
    try:
        return file_sha256(__file__)
    except Exception:
        return "unavailable"


# ═══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Patch-Lite Pangu distillation with decomposed loss.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="conf/config.yaml")
    p.add_argument("--teacher-checkpoint", required=True)
    p.add_argument("--student-init", "--student-init-checkpoint", default="")
    p.add_argument("--resume-checkpoint", default="")
    p.add_argument("--resume-optimizer", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--resume-world-size",
        type=int,
        default=None,
        help=(
            "World size used by the resumed scheduler. When it differs from the "
            "current DDP world size, scheduler step counters are rescaled to keep "
            "the learning-rate position continuous in epoch/sample space."
        ),
    )
    p.add_argument("--output-dir", default="checkpoints/e128_2x8x8_distill_v2")
    p.add_argument("--device", default="cuda:0")

    # Architecture
    p.add_argument("--patch-size", dest="patch_size", nargs=3, type=int, default=[2, 8, 8])
    p.add_argument("--embed-dim", dest="embed_dim", type=int, default=192)
    p.add_argument("--num-heads", dest="num_heads", nargs=4, type=int, default=[6, 12, 12, 6])
    p.add_argument("--teacher-patch-size", nargs=3, type=int, default=[2, 4, 4])
    p.add_argument("--teacher-embed-dim", type=int, default=192)
    p.add_argument("--teacher-num-heads", nargs=4, type=int, default=[6, 12, 12, 6])
    p.add_argument("--pad-student-input", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--residual-output", action="store_true", default=False,
                   help="Make the student backbone predict delta and return persistence_base + scale * delta.")
    p.add_argument("--residual-scale", type=float, default=1.0,
                   help="Scale applied to the network delta when --residual-output is enabled.")
    p.add_argument("--learnable-residual-scale", action="store_true", default=False,
                   help="Use one scalar trainable parameter for residual scale. Requires --residual-output.")

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Decomposed loss
    p.add_argument("--loss-preset", choices=list(LOSS_PRESETS.keys()), default=None,
                   help="Loss preset. If None, auto-detected from deprecated args.")
    p.add_argument("--w-all-teacher", type=float, default=None)
    p.add_argument("--w-15-teacher", type=float, default=None)
    p.add_argument("--w-15-gt", type=float, default=None)
    p.add_argument("--w-all-gt", type=float, default=None)
    p.add_argument("--official15-indices", nargs="+", type=int, default=None)

    # Deprecated
    p.add_argument("--lambda-distill", type=float, default=None,
                   help="[DEPRECATED] Auto-switches to legacy_weighted_all69.")
    p.add_argument("--lambda-gt", type=float, default=None,
                   help="[DEPRECATED] Auto-switches to legacy_weighted_all69.")
    p.add_argument("--official15-weight", type=float, default=None,
                   help="[DEPRECATED] Auto-switches to legacy_weighted_all69.")
    p.add_argument("--pressure-focus-levels", nargs="*", type=int, default=[850, 700, 500])
    p.add_argument("--pressure-focus-weight", type=float, default=2.0)

    # Feature distillation
    p.add_argument("--feature-distill", action="store_true", default=False)
    p.add_argument("--feature-stages", nargs="+", default=["layer2", "layer3"], choices=STAGE_ORDER)
    p.add_argument("--alpha-feature", type=float, default=0.0)

    # Init mode
    p.add_argument("--init-mode", choices=INIT_MODES, default="teacher_slice_interpolate")
    p.add_argument("--min-inherited-ratio", type=float, default=0.30)
    p.add_argument("--allow-low-inheritance", action="store_true")

    # Checkpoint dtype
    p.add_argument("--param-only-dtype", choices=("fp32", "fp16", "bf16"), default="fp32")

    # Resume protection
    p.add_argument("--allow-loss-change-on-resume", action="store_true")
    p.add_argument("--allow-legacy-resume", action="store_true")
    p.add_argument("--allow-nonempty-output-dir", action="store_true")

    # Reproducibility
    p.add_argument("--seed", type=int, default=20260704)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--fixed-overfit-subset", action="store_true")
    p.add_argument("--overfit-num-samples", type=int, default=None)
    p.add_argument("--sanity-only", action="store_true",
                   help="Run one no-grad forward/loss/shape check and exit without training or checkpointing.")

    # Infra
    p.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--max-train-steps", type=int, default=None)
    p.add_argument("--max-val-steps", type=int, default=None)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--dist-backend", default="nccl")
    p.add_argument("--local-rank", "--local_rank", type=int, default=None)
    p.add_argument("--ddp-find-unused-parameters", action="store_true")
    return p.parse_args()


FEATURE_DISTILL_DISABLED_MESSAGE = (
    "feature distillation is not implemented safely for Pangu token topology."
)


def validate_feature_distillation(args: argparse.Namespace) -> None:
    """Reject the incomplete token-feature path before any distributed or file setup."""
    if args.feature_distill:
        raise RuntimeError(FEATURE_DISTILL_DISABLED_MESSAGE)


def validate_fixed_overfit_config(args: argparse.Namespace) -> None:
    """Fail before data/model setup when a fixed-subset request is ambiguous."""
    if args.fixed_overfit_subset and (
        args.overfit_num_samples is None or args.overfit_num_samples <= 0
    ):
        raise ValueError(
            "--fixed-overfit-subset requires --overfit-num-samples to be a positive integer"
        )


BASE_CHANNEL_MAPPING = {
    "input_surface": [0, 1, 2, 3],
    "input_static_mask": [4, 5, 6],
    "input_upper": list(range(7, 72)),
    "output_surface": [0, 1, 2, 3],
    "output_upper": list(range(4, 69)),
    "expression": "torch.cat([x[:, :4], x[:, 7:]], dim=1)",
}


def residual_output_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "residual_output": bool(args.residual_output),
        "residual_scale": float(args.residual_scale),
        "learnable_residual_scale": bool(args.learnable_residual_scale),
        "base_channel_mapping": BASE_CHANNEL_MAPPING,
    }


def validate_residual_config(args: argparse.Namespace) -> None:
    if args.learnable_residual_scale and not args.residual_output:
        raise ValueError("--learnable-residual-scale requires --residual-output")


def attach_learnable_residual_scale(model: nn.Module, args: argparse.Namespace) -> None:
    if args.residual_output and args.learnable_residual_scale:
        model.residual_scale_param = nn.Parameter(
            torch.tensor(float(args.residual_scale), dtype=torch.float32)
        )


def validate_output_directory(
    output_dir: str | Path,
    resume_checkpoint: str,
    allow_nonempty: bool,
) -> None:
    """Prevent unrelated histories/checkpoints from being mixed by default."""
    path = Path(output_dir)
    if not path.exists() or resume_checkpoint or allow_nonempty:
        return
    if any(path.iterdir()):
        raise RuntimeError(
            f"output directory is not empty: {path}. Use --resume-checkpoint for a "
            "real resume or --allow-nonempty-output-dir to override explicitly."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Loss config resolution
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_loss_config(args: argparse.Namespace) -> dict[str, Any]:
    deprecated_warnings: list[str] = []
    old_args_used = any(x is not None for x in [args.lambda_distill, args.lambda_gt, args.official15_weight])

    # Auto-detect legacy
    if args.loss_preset is None and old_args_used:
        preset_name = "legacy_weighted_all69"
        deprecated_warnings.append(
            "Old loss args detected without --loss-preset. "
            "Auto-switching to legacy_weighted_all69."
        )
    elif args.loss_preset is None:
        preset_name = "teacher15"
    else:
        preset_name = args.loss_preset

    preset = dict(LOSS_PRESETS[preset_name])
    is_legacy = preset_name == "legacy_weighted_all69"

    # Map deprecated args
    lambda_distill = args.lambda_distill if args.lambda_distill is not None else 1.0
    lambda_gt = args.lambda_gt if args.lambda_gt is not None else 0.0
    official15_weight = args.official15_weight if args.official15_weight is not None else 2.0

    if is_legacy:
        preset["w_all_teacher"] = lambda_distill
        preset["w_all_gt"] = lambda_gt
        if args.official15_weight is not None:
            deprecated_warnings.append(
                f"--official15-weight={args.official15_weight} used in legacy mode."
            )
        if args.lambda_distill is not None:
            deprecated_warnings.append(
                f"--lambda-distill={args.lambda_distill} → w_all_teacher={lambda_distill} (legacy)."
            )
        if args.lambda_gt is not None:
            deprecated_warnings.append(
                f"--lambda-gt={args.lambda_gt} → w_all_gt={lambda_gt} (legacy)."
            )
    else:
        if old_args_used:
            for name, val in [("--lambda-distill", args.lambda_distill),
                              ("--lambda-gt", args.lambda_gt),
                              ("--official15-weight", args.official15_weight)]:
                if val is not None:
                    deprecated_warnings.append(
                        f"{name}={val} is DEPRECATED and ignored when using --loss-preset {preset_name}."
                    )

    # Explicit w_* overrides
    if args.w_all_teacher is not None:
        preset["w_all_teacher"] = args.w_all_teacher
    if args.w_15_teacher is not None:
        preset["w_15_teacher"] = args.w_15_teacher
    if args.w_15_gt is not None:
        preset["w_15_gt"] = args.w_15_gt
    if args.w_all_gt is not None:
        preset["w_all_gt"] = args.w_all_gt

    official15_indices = list(args.official15_indices or DEFAULT_OFFICIAL_15)
    has_gt_15 = preset["w_15_gt"] > 0
    has_gt_all = preset["w_all_gt"] > 0
    teacher_only = not has_gt_15 and not has_gt_all
    if has_gt_15 and has_gt_all:
        gt_scope = "mixed"
    elif has_gt_15:
        gt_scope = "official15"
    elif has_gt_all:
        gt_scope = "all69"
    else:
        gt_scope = "none"

    return {
        "loss_preset": preset_name,
        "is_legacy": is_legacy,
        "w_all_teacher": preset["w_all_teacher"],
        "w_15_teacher": preset["w_15_teacher"],
        "w_15_gt": preset["w_15_gt"],
        "w_all_gt": preset["w_all_gt"],
        "alpha_feature": args.alpha_feature,
        "official15_indices": official15_indices,
        "official15_weight": official15_weight,
        "teacher_only": teacher_only,
        "gt_used": not teacher_only,
        "gt_scope": gt_scope,
        "deprecated_warnings": deprecated_warnings,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════

def set_random_seed(seed: int | None, deterministic: bool = False) -> None:
    if seed is None:
        return
    pyrandom.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════════

def amp_dtype(name: str) -> torch.dtype | None:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return None


def autocast_context(device, dtype, enabled):
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def unwrap_state(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt


def filtered_state(state):
    return {k: v for k, v in state.items()
            if not any(p in k for p in DROP_PARAM_ONLY_PATTERNS)}


def padded_img_size(img_size, patch_size):
    h, w = list(img_size)
    _, p_h, p_w = list(patch_size)
    return [((h + p_h - 1) // p_h) * p_h, ((w + p_w - 1) // p_w) * p_w]


def symmetric_padding(cur, tgt):
    h, w = list(cur)
    th, tw = list(tgt)
    ph, pw = th - h, tw - w
    if ph < 0 or pw < 0:
        raise ValueError(f"input {(h,w)} > target {list(tgt)}")
    t, b = ph // 2, ph - ph // 2
    l, r = pw // 2, pw - pw // 2
    return l, r, t, b


def pad_to_img_size(x, img_size):
    h, w = x.shape[-2:]
    l, r, t, b = symmetric_padding((h, w), img_size)
    if l == r == t == b == 0:
        return x
    return F.pad(x, (l, r, t, b), mode="replicate")


def crop_to_img_size(x, img_size):
    th, tw = list(img_size)
    h, w = x.shape[-2:]
    l, _, t, _ = symmetric_padding(img_size, (h, w))
    return x[..., t:t + th, l:l + tw]


# ═══════════════════════════════════════════════════════════════════════════════
# Model building
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(cfg, img_size, patch_size, embed_dim, num_heads, device, retarget_recovery):
    img_size = list(img_size)
    patch_size = list(patch_size)
    model = Pangu(
        img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
        num_heads=list(num_heads), window_size=cfg.window_size,
    )
    if retarget_recovery:
        model.patchrecovery2d = OneRecovery(
            style="PanguPatchRecovery", img_size=img_size,
            patch_size=patch_size[1:], in_chans=embed_dim * 2, out_chans=4,
        )
        model.patchrecovery3d = OneRecovery(
            style="PanguPatchRecovery", img_size=(13, *img_size),
            patch_size=patch_size, in_chans=embed_dim * 2, out_chans=5,
        )
    return model.to(device)


def load_teacher(cfg, checkpoint, img_size, patch_size, embed_dim, num_heads, device):
    teacher = build_model(cfg, img_size, patch_size, embed_dim, num_heads, device, retarget_recovery=False)
    state = filtered_state(unwrap_state(checkpoint))
    load_state_dict_checked(
        teacher,
        state,
        context="teacher",
        allowed_missing_patterns=DROP_PARAM_ONLY_PATTERNS,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


# ═══════════════════════════════════════════════════════════════════════════════
# Student initialization
# ═══════════════════════════════════════════════════════════════════════════════

def _legacy_classify_param(key: str, shape: list[int]) -> str:
    """Deprecated one-axis classifier retained only for checkpoint archaeology."""
    if "patchembedding" in key or "patchrecovery" in key:
        return "patch_kernel"
    if "norm" in key or "Norm" in key:
        return "norm"
    if "bias" in key:
        return "bias"
    if len(shape) == 2:
        return "linear"
    if len(shape) >= 3:
        return "conv"
    return "other"


def _legacy_slice_dim(tensor: torch.Tensor, src_dim_size: int, dst_dim_size: int, dim: int) -> torch.Tensor:
    """Slice a tensor along `dim` from src_dim_size to dst_dim_size."""
    indices = list(range(dst_dim_size))
    return tensor.index_select(dim, torch.tensor(indices, device=tensor.device)).contiguous()


def init_random(student: Pangu) -> dict:
    return {
        "mode": "random",
        "inherited": [], "sliced": [], "interpolated": [], "skipped": [],
        "inherited_params": 0, "total_params": sum(p.numel() for p in student.parameters()),
        "inherited_ratio": 0.0,
        "warnings": ["HIGH RISK: random initialization — only for smoke testing."],
    }


def init_shape_match(student: Pangu, teacher_state: dict) -> dict:
    own = student.state_dict()
    inherited = {}
    skipped = []
    for key, value in filtered_state(teacher_state).items():
        if key in own and own[key].shape == value.shape:
            inherited[key] = value.float()
        else:
            skipped.append(f"{key} (shape {list(value.shape)} → student {list(own.get(key, torch.empty(0)).shape)})")
    student.load_state_dict(inherited, strict=False)
    total = sum(p.numel() for p in student.parameters())
    inh_params = sum(v.numel() for v in inherited.values())
    return {
        "mode": "shape_match",
        "inherited": list(inherited.keys()),
        "sliced": [], "interpolated": [], "skipped": skipped,
        "inherited_params": inh_params, "total_params": total,
        "inherited_ratio": inh_params / max(total, 1),
        "warnings": ["weak inheritance: only exact shape matches"],
    }


def _legacy_init_teacher_slice(student: Pangu, teacher_state: dict, teacher_embed: int, student_embed: int,
                               teacher_num_heads: list[int], student_num_heads: list[int]) -> dict:
    raise RuntimeError("legacy one-axis teacher slicing is disabled")
    own = student.state_dict()
    ts = filtered_state(teacher_state)
    inherited, sliced, skipped = {}, [], []
    te = teacher_embed
    se = student_embed

    # Map stage channel multipliers: layer1=1x, layer2=2x, layer3=2x, layer4=1x
    stage_mult = {"layer1": 1, "layer2": 2, "layer3": 2, "layer4": 1}

    for key, value in ts.items():
        if key not in own:
            skipped.append(f"{key} (not in student)")
            continue
        t_shape = list(value.shape)
        s_shape = list(own[key].shape)
        if t_shape == s_shape:
            inherited[key] = value.float()
            continue

        kind = _classify_param(key, t_shape)
        sliced_ok = False

        if kind in ("linear", "norm", "bias"):
            # Try to find the embed dim and slice
            new_val = value.float()
            for dim_idx in range(len(t_shape)):
                if t_shape[dim_idx] == te:
                    new_val = _slice_dim(new_val, te, se, dim_idx)
                    sliced_ok = True
                    break
                # Check for 2x embed (downsample/upsample)
                for mult in [2]:
                    if t_shape[dim_idx] == te * mult:
                        new_val = _slice_dim(new_val, te * mult, se * mult, dim_idx)
                        sliced_ok = True
                        break
                if sliced_ok:
                    break
            if sliced_ok and list(new_val.shape) == s_shape:
                sliced.append(f"{key} ({t_shape} → {s_shape})")
                inherited[key] = new_val
            else:
                skipped.append(f"{key} ({t_shape} → {s_shape}, cannot slice)")
        elif kind == "patch_kernel":
            skipped.append(f"{key} ({t_shape} → {s_shape}, patch_kernel — needs interpolation)")
        else:
            # Try generic: find any dim matching teacher embed
            new_val = value.float()
            found = False
            for dim_idx in range(len(t_shape)):
                if t_shape[dim_idx] == te:
                    new_val = _slice_dim(new_val, te, se, dim_idx)
                    found = True
                    break
                for mult in [2]:
                    if t_shape[dim_idx] == te * mult:
                        new_val = _slice_dim(new_val, te * mult, se * mult, dim_idx)
                        found = True
                        break
                if found:
                    break
            if found and list(new_val.shape) == s_shape:
                sliced.append(f"{key} ({t_shape} → {s_shape})")
                inherited[key] = new_val
            else:
                skipped.append(f"{key} ({t_shape} → {s_shape}, shape mismatch)")

    student.load_state_dict(inherited, strict=False)
    total = sum(p.numel() for p in student.parameters())
    inh_params = sum(v.numel() for v in inherited.values())
    ratio = inh_params / max(total, 1)
    warns = []
    if ratio < 0.3:
        warns.append(f"LOW INHERITANCE: {ratio:.1%} < 30% — model may be poorly initialized.")
    return {
        "mode": "teacher_slice",
        "inherited": [k for k in inherited if k not in [s.split(" (")[0] for s in sliced]],
        "sliced": sliced, "interpolated": [], "skipped": skipped,
        "inherited_params": inh_params, "total_params": total,
        "inherited_ratio": ratio,
        "warnings": warns,
    }


def _legacy_init_teacher_slice_interpolate(student: Pangu, teacher_state: dict,
                                            teacher_embed: int, student_embed: int,
                                            teacher_patch: list[int], student_patch: list[int],
                                            teacher_num_heads: list[int], student_num_heads: list[int]) -> dict:
    raise RuntimeError("legacy one-axis teacher slicing is disabled")
    report = _legacy_init_teacher_slice(student, teacher_state, teacher_embed, student_embed,
                                teacher_num_heads, student_num_heads)
    report["mode"] = "teacher_slice_interpolate"

    own = student.state_dict()
    ts = filtered_state(teacher_state)
    interpolated = []
    tp = list(teacher_patch)
    sp = list(student_patch)

    for key, value in ts.items():
        if key not in own:
            continue
        t_shape = list(value.shape)
        s_shape = list(own[key].shape)
        kind = _classify_param(key, t_shape)
        if kind != "patch_kernel":
            continue
        if t_shape == s_shape:
            continue  # already inherited in slice pass

        # Patch kernel: typically [out_ch, in_ch, D, H, W] for Conv3d/ConvTranspose3d
        # We need to handle spatial dim changes and channel slicing
        try:
            new_val = _interpolate_patch_kernel(value.float(), t_shape, s_shape, tp, sp,
                                                 teacher_embed, student_embed)
            if new_val is not None and list(new_val.shape) == s_shape:
                own_state = student.state_dict()
                own_state[key] = new_val
                student.load_state_dict(own_state, strict=False)
                interpolated.append(f"{key} ({t_shape} → {s_shape}, avg_pool+slice)")
        except Exception as e:
            report["skipped"].append(f"{key} ({t_shape} → {s_shape}, interp failed: {e})")

    # Re-count
    total = sum(p.numel() for p in student.parameters())
    current_state = student.state_dict()
    inh_params = 0
    for key in list(report.get("inherited", [])) + [s.split(" (")[0] for s in report.get("sliced", [])]:
        if key in current_state:
            inh_params += current_state[key].numel()
    for s in interpolated:
        key = s.split(" (")[0]
        if key in current_state:
            inh_params += current_state[key].numel()

    report["interpolated"] = interpolated
    report["inherited_params"] = inh_params
    report["total_params"] = total
    report["inherited_ratio"] = inh_params / max(total, 1)
    if report["inherited_ratio"] < 0.3:
        report["warnings"].append(f"LOW INHERITANCE: {report['inherited_ratio']:.1%} < 30%")
    return report


def _legacy_interpolate_patch_kernel(value: torch.Tensor, t_shape, s_shape, tp, sp,
                               te, se) -> torch.Tensor | None:
    """Interpolate a patch embedding/recovery kernel from teacher to student spatial dims."""
    if len(t_shape) < 3 or len(s_shape) < 3:
        return None

    # Determine spatial dims (last 3 for Conv3d, last 2 for Conv2d-like)
    spatial_start = 2 if len(t_shape) >= 5 else (2 if len(t_shape) >= 4 else None)
    if spatial_start is None:
        return None

    n_spatial = len(t_shape) - spatial_start
    if n_spatial == 3:
        t_d, t_h, t_w = t_shape[2], t_shape[3], t_shape[4]
        s_d, s_h, s_w = s_shape[2], s_shape[3], s_shape[4]
    elif n_spatial == 2:
        t_h, t_w = t_shape[2], t_shape[3]
        s_h, s_w = s_shape[2], s_shape[3]
        t_d, s_d = 1, 1
    else:
        return None

    # Step 1: Slice output channels (dim 0) if needed
    if t_shape[0] != s_shape[0]:
        out_slice = min(t_shape[0], s_shape[0])
        value = value[:out_slice]
        if out_slice < s_shape[0]:
            # Pad remaining with small random
            pad_shape = list(value.shape)
            pad_shape[0] = s_shape[0] - out_slice
            pad = torch.randn(pad_shape, dtype=value.dtype) * 0.01
            value = torch.cat([value, pad], dim=0)

    # Step 2: Slice input channels (dim 1) if needed
    if t_shape[1] != s_shape[1]:
        in_slice = min(t_shape[1], s_shape[1])
        value = value[:, :in_slice]
        if in_slice < s_shape[1]:
            pad_shape = list(value.shape)
            pad_shape[1] = s_shape[1] - in_slice
            pad = torch.randn(pad_shape, dtype=value.dtype) * 0.01
            value = torch.cat([value, pad], dim=1)

    # Step 3: Spatial interpolation
    if n_spatial == 3:
        # Use 3D average pooling then interpolate
        if value.ndim == 5:
            # [out, in, D, H, W]
            if t_d != s_d or t_h != s_h or t_w != s_w:
                out_ch, in_ch = value.shape[0], value.shape[1]
                # Reshape to [out_ch*in_ch, 1, D, H, W] for interpolation
                v = value.reshape(out_ch * in_ch, 1, t_d, t_h, t_w)
                v = F.interpolate(v, size=(s_d, s_h, s_w), mode="trilinear", align_corners=False)
                value = v.reshape(out_ch, in_ch, s_d, s_h, s_w)
    elif n_spatial == 2 and value.ndim >= 4:
        out_ch, in_ch = value.shape[0], value.shape[1]
        extra_dims = value.ndim - 2
        if extra_dims == 2:
            v = value.reshape(out_ch * in_ch, 1, t_shape[2], t_shape[3])
            v = F.interpolate(v, size=(s_shape[2], s_shape[3]), mode="bilinear", align_corners=False)
            value = v.reshape(out_ch, in_ch, s_shape[2], s_shape[3])

    return value


# V2 safe teacher-derived initialization. These definitions intentionally replace the
# legacy one-dimension slicer above while keeping old checkpoint compatibility helpers.
def _classify_param(key: str, shape: list[int]) -> str:
    lower = key.lower()
    if ("patchembed" in lower or "patchrecovery" in lower) and len(shape) >= 4:
        return "patch_kernel"
    if "earth_position_bias_table" in lower:
        return "earth_position_bias"
    if ".qkv." in lower or lower.endswith("qkv.weight") or lower.endswith("qkv.bias"):
        return "qkv"
    if "norm" in lower:
        return "norm"
    if "bias" in lower:
        return "bias"
    if len(shape) == 2:
        return "linear"
    if len(shape) >= 3:
        return "conv"
    return "other"


def _slice_tensor_to_shape(tensor: torch.Tensor, target_shape: Iterable[int]) -> torch.Tensor:
    """Deterministically take the leading slice in every changed dimension."""
    target = tuple(int(v) for v in target_shape)
    if tensor.ndim != len(target):
        raise ValueError(f"rank mismatch: source={list(tensor.shape)}, target={list(target)}")
    if any(dst > src for src, dst in zip(tensor.shape, target)):
        raise ValueError(f"cannot enlarge by slicing: source={list(tensor.shape)}, target={list(target)}")
    return tensor[tuple(slice(0, dst) for dst in target)].contiguous()


def _slice_qkv_tensor(tensor: torch.Tensor, target_shape: Iterable[int]) -> torch.Tensor:
    """Slice Q, K and V independently so their row segments remain aligned."""
    target = tuple(int(v) for v in target_shape)
    if tensor.ndim not in (1, 2) or tensor.shape[0] % 3 or target[0] % 3:
        raise ValueError(f"invalid QKV shape: source={list(tensor.shape)}, target={list(target)}")
    target_part_shape = (target[0] // 3, *target[1:])
    return torch.cat(
        [_slice_tensor_to_shape(part, target_part_shape) for part in tensor.chunk(3, dim=0)],
        dim=0,
    ).contiguous()


def _interpolate_patch_kernel(
    tensor: torch.Tensor,
    target_shape: Iterable[int],
) -> torch.Tensor:
    """Slice channels, resize spatial kernel and preserve each kernel's total scale."""
    target = tuple(int(v) for v in target_shape)
    if tensor.ndim not in (4, 5) or len(target) != tensor.ndim:
        raise ValueError(f"unsupported patch kernel rank: {tensor.ndim} -> {len(target)}")
    if target[0] > tensor.shape[0] or target[1] > tensor.shape[1]:
        raise ValueError("patch channel enlargement is not a teacher-derived slice")

    value = tensor[: target[0], : target[1]].float().contiguous()
    source_spatial = tuple(int(v) for v in value.shape[2:])
    target_spatial = target[2:]
    if source_spatial == target_spatial:
        return value
    channels = value.shape[0] * value.shape[1]
    reshaped = value.reshape(channels, 1, *source_spatial)
    mode = "trilinear" if tensor.ndim == 5 else "bilinear"
    resized = F.interpolate(reshaped, size=target_spatial, mode=mode, align_corners=False)
    scale = math.prod(source_spatial) / math.prod(target_spatial)
    return (resized * scale).reshape(*target).contiguous()


def _interpolate_earth_bias(
    tensor: torch.Tensor,
    target_shape: Iterable[int],
) -> torch.Tensor:
    """Slice attention heads and interpolate both earth-position table axes."""
    target = tuple(int(v) for v in target_shape)
    if tensor.ndim != 3 or len(target) != 3:
        raise ValueError(f"earth bias must be rank 3: {list(tensor.shape)} -> {list(target)}")
    if target[2] > tensor.shape[2]:
        raise ValueError("student earth-bias head count exceeds teacher head count")
    value = tensor[:, :, : target[2]].float()
    if tuple(value.shape[:2]) == target[:2]:
        return value.contiguous()
    table = value.permute(2, 0, 1).unsqueeze(1)
    table = F.interpolate(table, size=target[:2], mode="bilinear", align_corners=False)
    return table.squeeze(1).permute(1, 2, 0).contiguous()


def _init_teacher_derived(student: Pangu, teacher_state: dict, allow_interpolation: bool) -> dict:
    """Initialize safe student tensors solely from their full-model counterparts."""
    own = student.state_dict()
    teacher = filtered_state(teacher_state)
    loadable: dict[str, torch.Tensor] = {}
    exact: list[str] = []
    sliced: list[str] = []
    qkv_sliced: list[str] = []
    interpolated: list[str] = []
    skipped: list[str] = []

    for key, source in teacher.items():
        if key not in own:
            skipped.append(f"{key}: not present in student")
            continue
        source_shape = tuple(source.shape)
        target_shape = tuple(own[key].shape)
        if source_shape == target_shape:
            loadable[key] = source.float()
            exact.append(key)
            continue
        kind = _classify_param(key, list(source_shape))
        try:
            if kind == "qkv":
                mapped = _slice_qkv_tensor(source.float(), target_shape)
                category = qkv_sliced
                operation = "Q/K/V segmented slice"
            elif kind == "patch_kernel":
                if not allow_interpolation:
                    raise ValueError("spatial patch change requires interpolation mode")
                mapped = _interpolate_patch_kernel(source.float(), target_shape)
                category = interpolated
                operation = "channel slice + scale-preserving spatial interpolation"
            elif kind == "earth_position_bias":
                if not allow_interpolation:
                    raise ValueError("earth-position table change requires interpolation mode")
                mapped = _interpolate_earth_bias(source.float(), target_shape)
                category = interpolated
                operation = "head slice + table interpolation"
            else:
                mapped = _slice_tensor_to_shape(source.float(), target_shape)
                category = sliced
                operation = "all-dimension leading slice"
            if tuple(mapped.shape) != target_shape:
                raise ValueError(f"mapped shape is {list(mapped.shape)}")
            loadable[key] = mapped
            category.append(f"{key} ({list(source_shape)} -> {list(target_shape)}; {operation})")
        except (RuntimeError, ValueError) as exc:
            skipped.append(f"{key} ({list(source_shape)} -> {list(target_shape)}): {exc}")

    incompatible = student.load_state_dict(loadable, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"teacher-derived initialization unexpected keys: {incompatible.unexpected_keys[:20]}")
    parameter_names = {name for name, _ in student.named_parameters()}
    inherited_params = sum(loadable[key].numel() for key in loadable if key in parameter_names)
    total_params = sum(param.numel() for param in student.parameters())
    ratio = inherited_params / max(total_params, 1)
    return {
        "mode": "teacher_slice_interpolate" if allow_interpolation else "teacher_slice",
        "init_source": "full_teacher_checkpoint",
        "inherited": exact,
        "exact": exact,
        "sliced": sliced,
        "qkv_sliced": qkv_sliced,
        "interpolated": interpolated,
        "skipped": skipped,
        "inherited_params": inherited_params,
        "total_params": total_params,
        "inherited_ratio": ratio,
        "warnings": [],
    }


def init_teacher_slice(student: Pangu, teacher_state: dict, teacher_embed: int, student_embed: int,
                       teacher_num_heads: list[int], student_num_heads: list[int]) -> dict:
    del teacher_embed, student_embed, teacher_num_heads, student_num_heads
    return _init_teacher_derived(student, teacher_state, allow_interpolation=False)


def init_teacher_slice_interpolate(student: Pangu, teacher_state: dict,
                                    teacher_embed: int, student_embed: int,
                                    teacher_patch: list[int], student_patch: list[int],
                                    teacher_num_heads: list[int], student_num_heads: list[int]) -> dict:
    del teacher_embed, student_embed, teacher_patch, student_patch
    del teacher_num_heads, student_num_heads
    return _init_teacher_derived(student, teacher_state, allow_interpolation=True)


def enforce_inheritance_threshold(
    report: dict,
    minimum_ratio: float,
    sanity_only: bool,
    allow_low: bool,
) -> None:
    ratio = float(report.get("inherited_ratio", 0.0))
    if ratio >= minimum_ratio:
        return
    message = f"LOW INHERITANCE: inherited ratio {ratio:.2%} < required {minimum_ratio:.2%}."
    report.setdefault("warnings", []).append(message)
    if sanity_only or allow_low:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return
    raise RuntimeError(
        message + " Refusing training; use --allow-low-inheritance only after explicit review."
    )


def generate_init_report(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Patch-Lite Student Initialization Report",
        "",
        f"**Mode:** {report['mode']}",
        f"**Inherited ratio:** {report['inherited_ratio']:.2%}",
        f"**Inherited params:** {report['inherited_params']:,}",
        f"**Total params:** {report['total_params']:,}",
        f"**Exact tensors:** {len(report.get('inherited', []))}",
        f"**Sliced tensors:** {len(report.get('sliced', []))}",
        f"**QKV sliced tensors:** {len(report.get('qkv_sliced', []))}",
        f"**Interpolated tensors:** {len(report.get('interpolated', []))}",
        f"**Skipped tensors:** {len(report.get('skipped', []))}",
        "",
        "## Exact inherited (same shape)",
        "",
    ]
    for k in report.get("inherited", []):
        lines.append(f"- `{k}`")

    lines += ["", "## Sliced tensors", ""]
    for s in report.get("sliced", []):
        lines.append(f"- `{s}`")

    lines += ["", "## QKV segmented slices", ""]
    for s in report.get("qkv_sliced", []):
        lines.append(f"- `{s}`")

    lines += ["", "## Interpolated tensors", ""]
    for s in report.get("interpolated", []):
        lines.append(f"- `{s}`")

    lines += ["", "## Skipped tensors", ""]
    for s in report.get("skipped", []):
        lines.append(f"- `{s}`")

    lines += ["", "## Warnings", ""]
    for w in report.get("warnings", []):
        lines.append(f"- ⚠ {w}")

    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports" / "patch_lite_init_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_student_with_init(cfg, teacher_state, student_img_size, args, device):
    student = build_model(cfg, student_img_size, args.patch_size, args.embed_dim,
                          args.num_heads, device, retarget_recovery=True)

    mode = args.init_mode
    if mode == "random":
        report = init_random(student)
    elif mode == "shape_match":
        report = init_shape_match(student, teacher_state)
    elif mode == "teacher_slice":
        report = init_teacher_slice(student, teacher_state,
                                    args.teacher_embed_dim, args.embed_dim,
                                    args.teacher_num_heads, args.num_heads)
    elif mode == "teacher_slice_interpolate":
        report = init_teacher_slice_interpolate(
            student, teacher_state,
            args.teacher_embed_dim, args.embed_dim,
            args.teacher_patch_size, args.patch_size,
            args.teacher_num_heads, args.num_heads,
        )
    else:
        raise ValueError(f"Unknown init mode: {mode}")

    return student, report


# ═══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_surface_mask(cfg_data, device):
    sd = cfg_data.dataset.static_dir
    lm = torch.from_numpy(np.load(os.path.join(sd, "land_mask.npy")).astype(np.float32))
    st = torch.from_numpy(np.load(os.path.join(sd, "soil_type.npy")).astype(np.float32))
    topo = torch.from_numpy(np.load(os.path.join(sd, "topography.npy")).astype(np.float32))
    topo = (topo - topo.mean()) / (topo.std(unbiased=False) + 1e-6)
    return torch.stack([lm, st, topo], dim=0).unsqueeze(0).to(device)


def get_nested_attr(obj, names, default=None):
    for name in names:
        cur = obj
        ok = True
        for part in name.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok:
            return cur
    return default


def normalize_loader_result(result):
    if isinstance(result, tuple):
        if len(result) == 0:
            raise ValueError("empty dataloader result")
        return result[0], result[1] if len(result) > 1 else None
    return result, None


def build_train_val_dataloaders(cfg_data, world_size=1):
    distributed = world_size > 1
    try:
        dp = ERA5Datapipe(params=cfg_data, distributed=distributed)
        tl, ts = normalize_loader_result(dp.train_dataloader())
        vl, vs = normalize_loader_result(dp.val_dataloader())
        return tl, ts, vl, vs
    except (TypeError, AttributeError) as exc:
        if is_main_process():
            print(f"ERA5Datapipe fallback: {type(exc).__name__}: {exc}")
    dd = cfg_data.dataset.data_dir
    ch = cfg_data.dataset.channels
    bs = cfg_data.dataloader.batch_size
    nw = cfg_data.dataloader.num_workers
    ty = get_nested_attr(cfg_data, ("dataset.train_time", "dataset.train_years", "dataset.train"), None)
    vy = get_nested_attr(cfg_data, ("dataset.valid_time", "dataset.validation_time", "dataset.val_time"), None)
    tp = ERA5Datapipe(dataset_dir=dd, used_variables=ch, used_years=ty, distributed=distributed, batch_size=bs, num_workers=nw)
    vp = ERA5Datapipe(dataset_dir=dd, used_variables=ch, used_years=vy, distributed=distributed, batch_size=bs, num_workers=nw)
    tl, ts = normalize_loader_result(tp.get_dataloader("train"))
    vr = None
    for mode in ("val", "valid", "validation"):
        try:
            vr = vp.get_dataloader(mode)
            break
        except (KeyError, ValueError, AttributeError):
            continue
    if vr is None:
        raise RuntimeError("could not build val dataloader")
    vl, vs = normalize_loader_result(vr)
    return tl, ts, vl, vs


def describe_batch_item(item):
    if torch.is_tensor(item):
        return f"Tensor shape={list(item.shape)} dtype={item.dtype}"
    if isinstance(item, (list, tuple)):
        return f"{type(item).__name__}(len={len(item)})"
    return type(item).__name__


def print_batch_structure(data):
    if isinstance(data, (list, tuple)):
        print("first batch structure:")
        for idx, item in enumerate(data):
            print(f"  data[{idx}]: {describe_batch_item(item)}")


def find_target_candidate(data):
    if not isinstance(data, (list, tuple)):
        return None
    for item in data[1:]:
        if torch.is_tensor(item) and item.ndim >= 4:
            return item
    return None


def make_input_and_target(data, surface_mask, device):
    invar = data[0]
    tc = find_target_candidate(data)
    target = tc.to(device=device, dtype=torch.float32) if tc is not None else None
    inv_s = invar[:, :4, :, :].to(device=device, dtype=torch.float32)
    inv_u = invar[:, 4:, :, :].to(device=device, dtype=torch.float32)
    bm = surface_mask.expand(inv_s.shape[0], -1, -1, -1)
    return torch.cat([inv_s, bm, inv_u], dim=1), target


def build_persistence_base(x: torch.Tensor, output_img_size: list[int] | None = None) -> torch.Tensor:
    if x.ndim != 4:
        raise RuntimeError(f"residual base expects BCHW input, got shape {list(x.shape)}")
    if x.shape[1] != 72:
        raise RuntimeError(f"residual base expects 72 input channels, got {x.shape[1]}")
    base = torch.cat([x[:, :4], x[:, 7:]], dim=1)
    if base.shape[1] != 69:
        raise RuntimeError(f"residual base must have 69 channels, got {base.shape[1]}")
    if output_img_size is not None:
        base = crop_to_img_size(base, output_img_size)
    return base


def get_residual_scale(
    model: nn.Module,
    residual_scale: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    unwrapped = getattr(model, "module", model)
    parameter = getattr(unwrapped, "residual_scale_param", None)
    if parameter is not None:
        return parameter.to(device=device, dtype=dtype)
    return torch.tensor(float(residual_scale), device=device, dtype=dtype)


def apply_residual_output(
    model: nn.Module,
    x: torch.Tensor,
    delta: torch.Tensor,
    output_img_size: list[int] | None,
    residual_output: bool,
    residual_scale: float,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if not residual_output:
        return delta, None, delta
    base = build_persistence_base(x, output_img_size).to(device=delta.device, dtype=delta.dtype)
    scale = get_residual_scale(model, residual_scale, delta.dtype, delta.device)
    prediction = base + scale * delta
    return prediction, base, delta


def forward_69(model, x, model_img_size=None, output_img_size=None,
               residual_output=False, residual_scale=1.0, return_parts=False):
    if model_img_size is not None:
        x = pad_to_img_size(x, model_img_size)
    out_s, out_u = model(x)
    out_u = out_u.reshape(x.shape[0], 65, x.shape[-2], x.shape[-1])
    delta = torch.cat([out_s, out_u], dim=1)
    if output_img_size is not None:
        delta = crop_to_img_size(delta, output_img_size)
    pred, base, delta = apply_residual_output(
        model, x, delta, output_img_size, residual_output, residual_scale
    )
    if return_parts:
        return pred, base, delta
    return pred


# ═══════════════════════════════════════════════════════════════════════════════
# Loss: weights and computation
# ═══════════════════════════════════════════════════════════════════════════════

def build_all69_weights(device, pressure_focus_levels, pressure_focus_weight,
                         apply_official15_boost=False, official15_weight=2.0,
                         official15_indices=None):
    weights = torch.ones(69, dtype=torch.float32, device=device)
    weights[0] = 1.50 * 0.25
    weights[1] = 0.77 * 0.25
    weights[2] = 0.66 * 0.25
    weights[3] = 3.00 * 0.25
    uvar = {"Z": (4, 3.00), "Q": (17, 0.60), "T": (30, 1.50), "U": (43, 0.77), "V": (56, 0.54)}
    fi = []
    for lv in pressure_focus_levels:
        if lv not in PRESSURE_LEVELS:
            raise ValueError(f"pressure focus level {lv} not in {PRESSURE_LEVELS}")
        fi.append(PRESSURE_LEVELS.index(lv))
    for _, (base, vw) in uvar.items():
        for li in range(13):
            w = vw * (pressure_focus_weight if li in fi else 1.0)
            weights[base + li] = w
    if apply_official15_boost and official15_indices:
        weights[list(official15_indices)] *= official15_weight
    summary = {
        "pressure_focus_levels": list(pressure_focus_levels),
        "pressure_focus_indices": fi,
        "pressure_focus_weight": pressure_focus_weight,
        "official15_boost": apply_official15_boost,
        "min_weight": float(weights.min().detach().cpu()),
        "max_weight": float(weights.max().detach().cpu()),
    }
    return weights.view(1, 69, 1, 1), summary


def weighted_l1(pred, target, weights):
    return (torch.abs(pred - target) * weights).mean()


def official15_l1(pred, target, indices):
    return F.l1_loss(pred[:, indices], target[:, indices])


def compute_decomposed_loss(student_pred, teacher_pred, target, all69_weights,
                             official15_indices, loss_cfg, feature_loss):
    zero = student_pred.new_zeros(())
    lt_all = weighted_l1(student_pred, teacher_pred.detach(), all69_weights)
    lt_15 = official15_l1(student_pred, teacher_pred.detach(), official15_indices)

    needs_gt = loss_cfg["w_15_gt"] > 0 or loss_cfg["w_all_gt"] > 0
    gt_available = target is not None and target.shape == student_pred.shape

    # Task 4: GT safety — error if GT needed but unavailable
    if needs_gt:
        if target is None:
            raise RuntimeError(
                f"GT loss required (w_15_gt={loss_cfg['w_15_gt']}, w_all_gt={loss_cfg['w_all_gt']}) "
                f"but target is None. Ensure dataloader provides GT targets."
            )
        if target.shape != student_pred.shape:
            raise RuntimeError(
                f"GT loss required but target shape {list(target.shape)} != "
                f"student_pred shape {list(student_pred.shape)}."
            )

    if gt_available:
        # Always compute GT diagnostics for validation/checkpoint selection. Their
        # contribution to training remains controlled exclusively by the weights.
        lg_15 = official15_l1(student_pred, target, official15_indices)
        lg_all = weighted_l1(student_pred, target, all69_weights)
    else:
        lg_15 = zero
        lg_all = zero

    total = (loss_cfg["w_all_teacher"] * lt_all + loss_cfg["w_15_teacher"] * lt_15
             + loss_cfg["w_15_gt"] * lg_15 + loss_cfg["w_all_gt"] * lg_all
             + loss_cfg["alpha_feature"] * feature_loss)

    components = {
        "total": float(total.detach().cpu()),
        "teacher_all": float(lt_all.detach().cpu()),
        "teacher_15": float(lt_15.detach().cpu()),
        "gt_15": float(lg_15.detach().cpu()),
        "gt_all": float(lg_all.detach().cpu()),
        "feature": float(feature_loss.detach().cpu()),
    }
    return total, components, gt_available


def ensure_finite_loss(loss: torch.Tensor, context: str) -> None:
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError(f"non-finite loss at {context}: {float(loss.detach().cpu())}")


def classify_gradient_norm(
    grad_norm_value: float,
    scaler_enabled: bool,
    consecutive_overflows: int,
    context: str,
    max_consecutive_overflows: int = 8,
) -> dict[str, Any]:
    """Distinguish recoverable AMP overflow from genuine gradient failure."""
    if math.isfinite(grad_norm_value):
        return {
            "finite": True,
            "recoverable": False,
            "consecutive_overflows": 0,
        }

    next_overflows = consecutive_overflows + 1
    if not scaler_enabled:
        raise FloatingPointError(
            f"non-finite gradient norm at {context}: {grad_norm_value}"
        )
    if next_overflows > max_consecutive_overflows:
        raise FloatingPointError(
            f"consecutive AMP overflows exceeded {max_consecutive_overflows} at "
            f"{context}: grad_norm={grad_norm_value}"
        )
    return {
        "finite": False,
        "recoverable": True,
        "consecutive_overflows": next_overflows,
    }


def amp_step_was_skipped(scale_before: float, scale_after: float) -> bool:
    """Return whether GradScaler backed off after skipping optimizer.step."""
    return float(scale_after) < float(scale_before)


def validate_prediction_shape(
    prediction: torch.Tensor,
    output_img_size: Iterable[int],
    expected_batch: int = 1,
) -> None:
    expected = (int(expected_batch), 69, *map(int, output_img_size))
    if tuple(prediction.shape) != expected:
        raise RuntimeError(
            f"output shape mismatch: got {list(prediction.shape)}, expected {list(expected)}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Channel name helpers
# ═══════════════════════════════════════════════════════════════════════════════

def fallback_channel_name(index):
    if index == 0: return "mean_sea_level_pressure"
    if index == 1: return "10m_u_component_of_wind"
    if index == 2: return "10m_v_component_of_wind"
    if index == 3: return "2m_temperature"
    for start, name in [(4,"geopotential"),(17,"specific_humidity"),(30,"temperature"),
                        (43,"u_component_of_wind"),(56,"v_component_of_wind")]:
        if start <= index < start + 13:
            return f"{name}_{PRESSURE_LEVELS[index - start]}"
    return f"unknown_{index}"


def official15_descriptions(indices, channel_names):
    names = list(channel_names or [])
    return [f"{i}: {names[i] if i < len(names) else fallback_channel_name(i)}" for i in indices]


# ═══════════════════════════════════════════════════════════════════════════════
# Feature distillation (with topology protection)
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureCapture:
    def __init__(self, model, stages):
        self.features = {}
        self.handles = []
        for stage in stages:
            mod = getattr(model, stage, None)
            if mod is None:
                raise ValueError(f"no stage {stage!r}")
            self.handles.append(mod.register_forward_hook(self._hook(stage)))
    def _hook(self, stage):
        def h(_m, _i, o):
            t = _first_tensor(o)
            if t is not None:
                self.features[stage] = t
        return h
    def clear(self):
        self.features.clear()
    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def _first_tensor(v):
    if torch.is_tensor(v): return v
    if isinstance(v, (list, tuple)):
        for i in v:
            f = _first_tensor(i)
            if f is not None: return f
    if isinstance(v, dict):
        for i in v.values():
            f = _first_tensor(i)
            if f is not None: return f
    return None


def stage_channels(embed_dim):
    return {"layer1": embed_dim, "layer2": embed_dim * 2, "layer3": embed_dim * 2, "layer4": embed_dim}


class FeatureProjectors(nn.Module):
    def __init__(self, stages, sc, tc):
        super().__init__()
        self.projectors = nn.ModuleDict({s: nn.Conv3d(sc[s], tc[s], 1) for s in stages})
    def forward(self, stage, x):
        return self.projectors[stage](x)


def _validate_feature_topology(feat, expected_channels, stage_name):
    """Validate that a feature can be reshaped to [B,C,D,H,W] preserving spatial topology."""
    if feat.ndim == 5 and feat.shape[1] == expected_channels:
        return True
    if feat.ndim == 5 and feat.shape[-1] == expected_channels:
        return True  # [B,D,H,W,C] can be permuted
    # Reject flat token sequences that can't be mapped to a grid
    if feat.ndim == 3:
        # [B, N, C] — cannot safely determine D,H,W without grid info
        raise RuntimeError(
            f"Feature distillation topology error: stage {stage_name} has shape {list(feat.shape)} "
            f"which is a flat token sequence [B,N,C]. Cannot safely reshape to [B,C,D,H,W] "
            f"without grid mapping. Feature distillation is disabled for safety."
        )
    return True


def compute_feature_loss(sf, tf, proj, stages, sc, tc, print_shapes):
    raise RuntimeError(FEATURE_DISTILL_DISABLED_MESSAGE)
    losses, slog = [], {}
    active = list(stages)
    tw = sum(DEFAULT_STAGE_WEIGHTS[s] for s in active)
    for stage in active:
        if stage not in sf or stage not in tf:
            continue
        # Validate topology before proceeding
        _validate_feature_topology(sf[stage], sc[stage], stage)
        _validate_feature_topology(tf[stage], tc[stage], stage)
        # ... (rest of feature loss logic, only reached if topology is valid)
    if not losses:
        return next(proj.parameters()).new_zeros(()), slog
    return sum(losses), slog


# ═══════════════════════════════════════════════════════════════════════════════
# LR Scheduler
# ═══════════════════════════════════════════════════════════════════════════════

class WarmupCosineStepScheduler:
    def __init__(self, optimizer, base_lr, min_lr, warmup_steps, total_steps, last_step=0):
        self.optimizer = optimizer
        self.base_lr, self.min_lr = float(base_lr), float(min_lr)
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.last_step = int(last_step)
        self._set(self.lr_at(self.last_step))
    def lr_at(self, step):
        step = max(0, int(step))
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.min_lr + (self.base_lr - self.min_lr) * step / self.warmup_steps
        ds = max(1, self.total_steps - self.warmup_steps)
        p = min(max(step - self.warmup_steps, 0) / ds, 1.0)
        return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * p))
    def step(self):
        self.last_step += 1
        self._set(self.lr_at(self.last_step))
    def _set(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr
    def state_dict(self):
        return {"base_lr": self.base_lr, "min_lr": self.min_lr,
                "warmup_steps": self.warmup_steps, "total_steps": self.total_steps,
                "last_step": self.last_step}
    def load_state_dict(self, state):
        for k in ("base_lr", "min_lr"):
            setattr(self, k, float(state.get(k, getattr(self, k))))
        for k in ("warmup_steps", "total_steps", "last_step"):
            setattr(self, k, int(state.get(k, getattr(self, k))))
        self._set(self.lr_at(self.last_step))


def rebase_scheduler_after_resume(
    scheduler: WarmupCosineStepScheduler,
    additional_steps: int,
) -> None:
    """Treat requested epochs/steps as additional work beyond the resumed step."""
    scheduler.total_steps = max(scheduler.last_step + max(1, int(additional_steps)), 1)
    scheduler.warmup_steps = min(scheduler.warmup_steps, scheduler.total_steps - 1)
    scheduler._set(scheduler.lr_at(scheduler.last_step))


def rescale_scheduler_for_world_size(
    scheduler: WarmupCosineStepScheduler,
    old_world_size: int,
    new_world_size: int,
) -> dict[str, Any]:
    """Convert scheduler counters while preserving sample/epoch progress."""
    if old_world_size <= 0 or new_world_size <= 0:
        raise ValueError("world sizes must be positive")
    if old_world_size == new_world_size:
        return {
            "rescaled": False,
            "old_world_size": old_world_size,
            "new_world_size": new_world_size,
        }

    ratio = float(old_world_size) / float(new_world_size)
    before = {
        "warmup_steps": scheduler.warmup_steps,
        "total_steps": scheduler.total_steps,
        "last_step": scheduler.last_step,
        "lr": scheduler.lr_at(scheduler.last_step),
    }
    scheduler.warmup_steps = max(0, round(scheduler.warmup_steps * ratio))
    scheduler.total_steps = max(1, round(scheduler.total_steps * ratio))
    scheduler.last_step = max(0, round(scheduler.last_step * ratio))
    scheduler._set(scheduler.lr_at(scheduler.last_step))
    return {
        "rescaled": True,
        "old_world_size": old_world_size,
        "new_world_size": new_world_size,
        "ratio": ratio,
        "before": before,
        "after": {
            "warmup_steps": scheduler.warmup_steps,
            "total_steps": scheduler.total_steps,
            "last_step": scheduler.last_step,
            "lr": scheduler.lr_at(scheduler.last_step),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint save / load / resume protection
# ═══════════════════════════════════════════════════════════════════════════════

def effective_steps(loader, max_steps):
    try:
        ls = len(loader)
    except TypeError:
        ls = max_steps if max_steps is not None else 1
    if max_steps is not None:
        return max(1, min(ls, max_steps))
    return max(1, ls)


def build_full_metadata(args, loss_cfg, cfg, student_img_size, output_img_size,
                         cfg_data, teacher_ckpt_sha256, init_report, seed):
    return {
        "script_sha256": script_sha256(),
        "created_at": datetime.datetime.now().isoformat(),
        "teacher_checkpoint_path": args.teacher_checkpoint,
        "teacher_checkpoint_sha256": teacher_ckpt_sha256,
        "student_arch": {
            "patch_size": args.patch_size, "embed_dim": args.embed_dim,
            "num_heads": args.num_heads, "window_size": list(cfg.window_size),
            "img_size": student_img_size,
        },
        "teacher_arch": {
            "patch_size": args.teacher_patch_size, "embed_dim": args.teacher_embed_dim,
            "num_heads": args.teacher_num_heads, "window_size": list(cfg.window_size),
        },
        "init_mode": init_report.get("mode", args.init_mode),
        "init_source": init_report.get("init_source", args.init_mode),
        "student_init_path": init_report.get("student_init_path", ""),
        "student_init_sha256": init_report.get("student_init_sha256", ""),
        "init_report_summary": {
            "inherited_count": len(init_report.get("inherited", [])),
            "sliced_count": len(init_report.get("sliced", [])),
            "qkv_sliced_count": len(init_report.get("qkv_sliced", [])),
            "interpolated_count": len(init_report.get("interpolated", [])),
            "skipped_count": len(init_report.get("skipped", [])),
            "inherited_params": init_report["inherited_params"],
            "total_params": init_report["total_params"],
            "inherited_ratio": init_report["inherited_ratio"],
        },
        "loss_config": {
            "loss_preset": loss_cfg["loss_preset"],
            "w_all_teacher": loss_cfg["w_all_teacher"],
            "w_15_teacher": loss_cfg["w_15_teacher"],
            "w_15_gt": loss_cfg["w_15_gt"],
            "w_all_gt": loss_cfg["w_all_gt"],
            "alpha_feature": loss_cfg["alpha_feature"],
            "official15_indices": loss_cfg["official15_indices"],
            "teacher_only": loss_cfg["teacher_only"],
            "gt_used": loss_cfg["gt_used"],
            "gt_scope": loss_cfg["gt_scope"],
        },
        "data_config": {
            "data_dir": str(getattr(cfg_data.dataset, "data_dir", "")),
            "stats_dir": str(getattr(cfg_data.dataset, "stats_dir", "")),
            "static_dir": str(getattr(cfg_data.dataset, "static_dir", "")),
        },
        "train_years": list(getattr(cfg_data.dataset, "train_ratio", [])),
        "val_years": list(getattr(cfg_data.dataset, "val_ratio", [])),
        "output_shape_contract": {"channels": 69, "img_size": output_img_size, "dtype": args.dtype},
        "save_dtype_contract": args.param_only_dtype,
        "official15_indices": loss_cfg["official15_indices"],
        **residual_output_metadata(args),
        "random_seed": seed,
    }


def load_state_dict_checked(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    context: str,
    allowed_missing_patterns: Iterable[str] = DROP_PARAM_ONLY_PATTERNS,
) -> torch.nn.modules.module._IncompatibleKeys:
    """Use strict=False only with an explicit, audited missing-key allowlist."""
    incompatible = model.load_state_dict(state, strict=False)
    patterns = tuple(allowed_missing_patterns)
    bad_missing = [
        key for key in incompatible.missing_keys
        if not any(pattern in key for pattern in patterns)
    ]
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"{context} state mismatch: bad_missing={bad_missing[:20]}, "
            f"unexpected={incompatible.unexpected_keys[:20]}"
        )
    return incompatible


def load_student_init_override(student: nn.Module, checkpoint: str | Path) -> dict:
    """Partially load a student checkpoint and return truthful override metadata."""
    path = Path(checkpoint)
    state = filtered_state(unwrap_state(path))
    own = student.state_dict()
    loadable: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in state.items():
        if key not in own:
            skipped.append(f"{key}: not present in student")
        elif own[key].shape != value.shape:
            skipped.append(f"{key}: shape {list(value.shape)} != {list(own[key].shape)}")
        else:
            loadable[key] = value.float()
    incompatible = student.load_state_dict(loadable, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"student-init unexpected keys: {incompatible.unexpected_keys[:20]}")
    parameter_names = {name for name, _ in student.named_parameters()}
    inherited_params = sum(loadable[key].numel() for key in loadable if key in parameter_names)
    total_params = sum(param.numel() for param in student.parameters())
    return {
        "mode": "student_init_override",
        "init_source": "student_init_override",
        "student_init_path": str(path),
        "student_init_sha256": file_sha256(path),
        "inherited": sorted(loadable),
        "exact": sorted(loadable),
        "sliced": [],
        "qkv_sliced": [],
        "interpolated": [],
        "skipped": skipped,
        "missing_after_partial_load": list(incompatible.missing_keys),
        "inherited_params": inherited_params,
        "total_params": total_params,
        "inherited_ratio": inherited_params / max(total_params, 1),
        "warnings": ["student-init overrides the teacher-derived initialization report"],
    }


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    """Write a checkpoint completely, fsync it, then atomically replace the destination."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def select_improved_checkpoints(
    val_total: float,
    val_teacher15: float,
    val_gt15: float,
    best_val_total: float,
    best_teacher15: float,
    best_gt15: float,
) -> dict[str, bool]:
    """Select all required checkpoints without coupling diagnostics to loss weights."""
    return {
        "teacher15": val_teacher15 < best_teacher15,
        "gt15": val_gt15 < best_gt15,
        "val_total": val_total < best_val_total,
    }


def should_save_full_checkpoint(
    fixed_overfit_subset: bool,
    epoch_number: int,
    final_epoch: int,
    save_every: int,
) -> bool:
    """Avoid rewriting optimizer-heavy checkpoints on every diagnostic epoch."""
    if not fixed_overfit_subset:
        return True
    return epoch_number == final_epoch or (
        save_every > 1 and epoch_number % save_every == 0
    )


def save_checkpoint(path, student, optimizer, scheduler, scaler, epoch, args,
                     best_val_loss, best_teacher15, best_gt15, fp, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": unwrap_model(student).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": vars(args),
        "best_val_loss": best_val_loss,
        "best_teacher15_loss": best_teacher15,
        "best_gt15_loss": best_gt15,
    }
    payload.update(metadata)
    if fp is not None:
        payload["feature_projector_state_dict"] = unwrap_model(fp).state_dict()
    atomic_torch_save(payload, path)


def save_param_only(path, student, metadata, dtype_str="fp32"):
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dt = dtype_map.get(dtype_str, torch.float32)
    state = {}
    for key, value in unwrap_model(student).state_dict().items():
        if any(p in key for p in DROP_PARAM_ONLY_PATTERNS):
            continue
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.is_floating_point():
                value = value.to(dt)
        state[key] = value
    payload = {"model_state_dict": state}
    # Include architecture metadata
    for mk in ("student_arch", "teacher_arch", "init_mode", "init_source",
               "student_init_path", "student_init_sha256", "init_report_summary",
               "loss_config", "teacher_checkpoint_sha256", "output_shape_contract",
               "save_dtype_contract", "teacher_checkpoint_path", "random_seed",
               "fixed_overfit_subset", "overfit_num_samples", "seed",
               "sample_digest", "train_sample_digest", "val_sample_digest",
               "residual_output", "residual_scale", "learnable_residual_scale",
               "base_channel_mapping"):
        if mk in metadata:
            payload[mk] = metadata[mk]
    payload["padding_config"] = metadata.get("student_arch", {}).get("img_size", [])
    payload["dtype"] = dtype_str
    atomic_torch_save(payload, path)


def normalize_arch_value(value):
    """Normalize sequence-valued architecture fields while preserving scalars."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def validate_resume(ckpt, args, loss_cfg, cfg_data, allow_loss_change, allow_legacy):
    """Validate checkpoint compatibility before resume."""
    meta = {k: v for k, v in ckpt.items() if k in (
        "student_arch", "teacher_checkpoint_sha256", "loss_config",
        "train_years", "val_years", "init_mode", "script_sha256",
    )}

    if not meta.get("student_arch"):
        if not allow_legacy:
            raise RuntimeError(
                "Checkpoint has no metadata. This is a legacy checkpoint. "
                "Use --allow-legacy-resume to load it (with warning)."
            )
        warnings.warn("Loading legacy checkpoint without metadata. No compatibility checks performed.")
        return

    # Check student arch
    cur_arch = {"patch_size": args.patch_size, "embed_dim": args.embed_dim,
                "num_heads": args.num_heads}
    ckpt_arch = meta["student_arch"]
    for k in ("patch_size", "embed_dim", "num_heads"):
        if normalize_arch_value(ckpt_arch.get(k)) != normalize_arch_value(cur_arch[k]):
            raise RuntimeError(f"Resume arch mismatch: {k} checkpoint={ckpt_arch.get(k)} vs current={cur_arch[k]}")

    # Check teacher hash
    if meta.get("teacher_checkpoint_sha256"):
        cur_hash = file_sha256(args.teacher_checkpoint) if os.path.exists(args.teacher_checkpoint) else ""
        if cur_hash and cur_hash != meta["teacher_checkpoint_sha256"]:
            raise RuntimeError(f"Resume teacher hash mismatch: {cur_hash[:16]} vs {meta['teacher_checkpoint_sha256'][:16]}")

    # Check loss config
    if not allow_loss_change and meta.get("loss_config"):
        ckpt_lc = meta["loss_config"]
        for k in ("w_all_teacher", "w_15_teacher", "w_15_gt", "w_all_gt"):
            if abs(ckpt_lc.get(k, 0) - loss_cfg[k]) > 1e-9:
                raise RuntimeError(
                    f"Resume loss_config mismatch: {k} checkpoint={ckpt_lc.get(k)} vs current={loss_cfg[k]}. "
                    f"Use --allow-loss-change-on-resume to override."
                )

    # Check init_mode
    if meta.get("init_mode") and meta["init_mode"] != args.init_mode:
        warnings.warn(f"Resume init_mode mismatch: {meta['init_mode']} vs {args.init_mode}")

    # Check data years
    for yr_key, attr in [("train_years", "train_ratio"), ("val_years", "val_ratio")]:
        ckpt_yr = meta.get(yr_key, [])
        cur_yr = list(getattr(cfg_data.dataset, attr, []))
        if ckpt_yr and cur_yr and ckpt_yr != cur_yr:
            warnings.warn(f"Resume {yr_key} mismatch: {ckpt_yr} vs {cur_yr}")


def load_resume(checkpoint, student, optimizer, scheduler, scaler, fp,
                resume_optimizer, lr, args, loss_cfg, cfg_data,
                allow_loss_change, allow_legacy):
    if not checkpoint:
        return 0, float("inf"), float("inf"), float("inf")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_resume(ckpt, args, loss_cfg, cfg_data, allow_loss_change, allow_legacy)
    load_state_dict_checked(
        student,
        filtered_state(ckpt["model_state_dict"]),
        context="student resume",
        allowed_missing_patterns=DROP_PARAM_ONLY_PATTERNS,
    )
    if fp is not None and "feature_projector_state_dict" in ckpt:
        load_state_dict_checked(
            fp,
            ckpt["feature_projector_state_dict"],
            context="feature projector resume",
            allowed_missing_patterns=(),
        )
    if resume_optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for g in optimizer.param_groups:
            g["lr"] = lr
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler.is_enabled() and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return (
        int(ckpt.get("epoch", 0)),
        float(ckpt.get("best_val_loss", float("inf"))),
        float(ckpt.get("best_teacher15_loss", ckpt.get("best_official15_loss", float("inf")))),
        float(ckpt.get("best_gt15_loss", float("inf"))),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Distributed helpers
# ═══════════════════════════════════════════════════════════════════════════════

def avg(rows, key):
    return sum(r.get(key, 0.0) for r in rows) / max(len(rows), 1)


def setup_distributed(args):
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws <= 1:
        return torch.device(args.device), 0, 0, 1
    lr = args.local_rank
    if lr is None:
        lr = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(lr)
        device = torch.device("cuda", lr)
    else:
        device = torch.device(args.device)
    dist.init_process_group(backend=args.dist_backend)
    return device, rank, lr, ws


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def unwrap_model(m):
    return m.module if hasattr(m, "module") else m


def set_sampler_epoch(sampler, epoch):
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def reduce_metrics(metrics, device):
    if not dist.is_available() or not dist.is_initialized():
        return metrics
    keys = sorted(metrics.keys())
    vals = torch.tensor([metrics[k] for k in keys], dtype=torch.float32, device=device)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    vals /= dist.get_world_size()
    return {k: float(v.item()) for k, v in zip(keys, vals)}


# ═══════════════════════════════════════════════════════════════════════════════
# Fixed overfit subset
# ═══════════════════════════════════════════════════════════════════════════════

class FixedSubsetSampler(torch.utils.data.Sampler):
    """Always yields the same fixed set of indices, ignoring epoch changes."""
    def __init__(self, indices, world_size=1, rank=0):
        self.indices = list(indices)
        self.world_size = world_size
        self.rank = rank
        # In DDP, each rank gets a disjoint shard
        if world_size > 1:
            self.indices = self.indices[self.rank::self.world_size]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        pass  # Intentionally ignore — fixed subset


def build_fixed_overfit_indices(loader, num_samples, seed, world_size):
    """Build fixed indices from the first num_samples of the loader."""
    try:
        total = len(loader.dataset)
    except (TypeError, AttributeError):
        total = num_samples
    if seed is not None:
        rng = np.random.RandomState(seed)
        indices = rng.choice(total, size=min(num_samples, total), replace=False).tolist()
    else:
        indices = list(range(min(num_samples, total)))
    return sorted(indices)


def fixed_subset_digest(indices, seed, dataset_size):
    """Hash a canonical fixed-subset description; never consumed by the model."""
    payload = {
        "dataset_size": int(dataset_size),
        "indices": [int(index) for index in indices],
        "seed": int(seed) if seed is not None else None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_fixed_subset_metadata(
    train_indices,
    val_indices,
    seed,
    requested_samples,
    train_dataset_size,
    val_dataset_size,
    world_size,
):
    """Build a global, rank-independent audit record for the diagnostic subset."""
    train_indices = [int(index) for index in train_indices]
    val_indices = [int(index) for index in val_indices]
    train_digest = fixed_subset_digest(train_indices, seed, train_dataset_size)
    val_digest = fixed_subset_digest(val_indices, seed, val_dataset_size)
    combined = json.dumps(
        {"train": train_digest, "val": val_digest},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "fixed_overfit_subset": True,
        "overfit_num_samples": int(requested_samples),
        "seed": int(seed) if seed is not None else None,
        "sample_digest": hashlib.sha256(combined).hexdigest(),
        "train_sample_digest": train_digest,
        "val_sample_digest": val_digest,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "train_dataset_size": int(train_dataset_size),
        "val_dataset_size": int(val_dataset_size),
        "world_size": int(world_size),
        "rank_sharding": "global_indices_then_stride_by_rank",
        "model_input_usage": False,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Startup print
# ═══════════════════════════════════════════════════════════════════════════════

def print_startup(args, cfg, loss_cfg, student_img_size, teacher_img_size,
                   output_img_size, device, amp_enabled, weight_summary, init_report, seed):
    print("=" * 72)
    print(f"student patch_size={args.patch_size}, embed_dim={args.embed_dim}, num_heads={args.num_heads}")
    print(f"teacher patch_size={args.teacher_patch_size}, embed_dim={args.teacher_embed_dim}, num_heads={args.teacher_num_heads}")
    print(f"student padded img_size={student_img_size}, teacher img_size={teacher_img_size}, output={output_img_size}")
    print(f"dtype={args.dtype}, amp={amp_enabled}, device={device}")
    print(f"init_mode={args.init_mode}, inherited_ratio={init_report['inherited_ratio']:.2%}")
    print(f"seed={seed}")
    print(f"loss_preset={loss_cfg['loss_preset']}, is_legacy={loss_cfg['is_legacy']}")
    print(f"  w_all_teacher={loss_cfg['w_all_teacher']}, w_15_teacher={loss_cfg['w_15_teacher']}")
    print(f"  w_15_gt={loss_cfg['w_15_gt']}, w_all_gt={loss_cfg['w_all_gt']}")
    print(f"  alpha_feature={loss_cfg['alpha_feature']}, gt_scope={loss_cfg['gt_scope']}")
    print(f"  official15_count={len(loss_cfg['official15_indices'])}")
    if loss_cfg["is_legacy"]:
        print(f"  official15_weight={loss_cfg['official15_weight']} (legacy boost on all69 weights)")
    print(f"feature_distill={args.feature_distill}, feature_stages={args.feature_stages}")
    print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

LOSS_KEYS = ["total", "teacher_all", "teacher_15", "gt_15", "gt_all", "feature"]


def main():
    args = parse_args()
    sys.path.append(str(Path.cwd()))
    validate_feature_distillation(args)
    validate_fixed_overfit_config(args)
    validate_residual_config(args)
    if not 0.0 <= args.min_inherited_ratio <= 1.0:
        raise ValueError("--min-inherited-ratio must be between 0 and 1")
    if int(os.environ.get("RANK", "0")) == 0:
        validate_output_directory(
            args.output_dir,
            resume_checkpoint=args.resume_checkpoint,
            allow_nonempty=args.allow_nonempty_output_dir,
        )

    # Resolve loss config BEFORE distributed setup (for logging)
    loss_cfg = resolve_loss_config(args)

    # Seed
    seed = args.seed
    set_random_seed(seed, args.deterministic)

    device, rank, local_rank, world_size = setup_distributed(args)
    main_process = is_main_process()
    out_dir = Path(args.output_dir)

    if main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        run_args = vars(args) | {
            "distributed": world_size > 1, "rank": rank,
            "local_rank": local_rank, "world_size": world_size,
            "resolved_loss_config": {k: v for k, v in loss_cfg.items() if k != "deprecated_warnings"},
        }
        (out_dir / "run_args.json").write_text(json.dumps(run_args, ensure_ascii=False, indent=2), encoding="utf-8")
        for w in loss_cfg["deprecated_warnings"]:
            warnings.warn(w, DeprecationWarning, stacklevel=2)
            print(f"⚠ DEPRECATED: {w}")
    barrier()

    cfg = YParams(args.config, "model")
    cfg_data = YParams(args.config, "datapipe")
    if args.num_workers is not None:
        cfg_data.dataloader.num_workers = args.num_workers

    train_loader, _train_sampler, val_loader, _val_sampler = build_train_val_dataloaders(cfg_data, world_size=world_size)

    # Fixed overfit subset
    fixed_train_sampler = None
    fixed_val_sampler = None
    fixed_subset_metadata = None
    if args.fixed_overfit_subset:
        n_train = args.overfit_num_samples
        n_val = min(n_train, 10)
        train_indices = build_fixed_overfit_indices(train_loader, n_train, seed, world_size)
        val_indices = build_fixed_overfit_indices(val_loader, n_val, seed, world_size)
        fixed_subset_metadata = build_fixed_subset_metadata(
            train_indices=train_indices,
            val_indices=val_indices,
            seed=seed,
            requested_samples=n_train,
            train_dataset_size=len(train_loader.dataset),
            val_dataset_size=len(val_loader.dataset),
            world_size=world_size,
        )
        fixed_train_sampler = FixedSubsetSampler(train_indices, world_size, rank)
        fixed_val_sampler = FixedSubsetSampler(val_indices, world_size, rank)
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset, batch_size=cfg_data.dataloader.batch_size,
            sampler=fixed_train_sampler, num_workers=cfg_data.dataloader.num_workers,
            pin_memory=False, drop_last=False,
        )
        val_loader = torch.utils.data.DataLoader(
            val_loader.dataset, batch_size=cfg_data.dataloader.batch_size,
            sampler=fixed_val_sampler, num_workers=cfg_data.dataloader.num_workers,
            pin_memory=False, drop_last=False,
        )
        if main_process:
            (out_dir / "fixed_subset_manifest.json").write_text(
                json.dumps(fixed_subset_metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"Fixed overfit: train={len(train_indices)} samples, "
                f"val={len(val_indices)} samples, seed={seed}, "
                f"digest={fixed_subset_metadata['sample_digest']}"
            )

    output_img_size = list(cfg_data.dataset.img_size)
    student_img_size = padded_img_size(output_img_size, args.patch_size) if args.pad_student_input else output_img_size
    teacher_img_size = output_img_size
    dtype = amp_dtype(args.dtype)
    amp_enabled = (not args.no_amp) and dtype is not None and device.type in ("cuda", "xpu")

    # Teacher checkpoint SHA256
    teacher_ckpt_sha256 = ""
    if main_process:
        try:
            teacher_ckpt_sha256 = file_sha256(args.teacher_checkpoint)
        except Exception:
            teacher_ckpt_sha256 = "unavailable"

    teacher_state = unwrap_state(args.teacher_checkpoint)
    teacher = load_teacher(cfg, args.teacher_checkpoint, teacher_img_size,
                           args.teacher_patch_size, args.teacher_embed_dim,
                           args.teacher_num_heads, device)

    # Build student with init
    student, init_report = build_student_with_init(cfg, teacher_state, student_img_size, args, device)
    if args.student_init:
        init_report = load_student_init_override(student, args.student_init)
        if main_process:
            print(
                f"Student init override loaded {len(init_report['inherited'])} tensors "
                f"({init_report['inherited_ratio']:.2%} of student parameters) from {args.student_init}"
            )
    elif args.init_mode == "teacher_slice_interpolate":
        enforce_inheritance_threshold(
            init_report,
            args.min_inherited_ratio,
            sanity_only=args.sanity_only,
            allow_low=args.allow_low_inheritance,
        )
    attach_learnable_residual_scale(student, args)

    if main_process:
        generate_init_report(init_report, out_dir)
        for w in init_report.get("warnings", []):
            print(f"⚠ INIT: {w}")

    # Feature distillation
    student_channels = stage_channels(args.embed_dim)
    teacher_channels = stage_channels(args.teacher_embed_dim)
    feature_projectors = None
    student_capture = None
    teacher_capture = None
    if args.feature_distill:
        if args.alpha_feature <= 0:
            if main_process:
                print("⚠ feature_distill=True but alpha_feature=0 — feature loss will have no effect.")
        feature_projectors = FeatureProjectors(args.feature_stages, student_channels, teacher_channels).to(device)
        student_capture = FeatureCapture(student, args.feature_stages)
        teacher_capture = FeatureCapture(teacher, args.feature_stages)

    trainable = list(student.parameters())
    if feature_projectors is not None:
        trainable += list(feature_projectors.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.min_lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    steps_per_epoch = effective_steps(train_loader, args.max_train_steps)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(0, args.warmup_epochs * steps_per_epoch)
    scheduler = WarmupCosineStepScheduler(optimizer, args.lr, args.min_lr, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and args.dtype == "fp16" and device.type == "cuda")

    # Resume with protection
    start_epoch, best_val_loss, best_teacher15, best_gt15 = load_resume(
        args.resume_checkpoint, student, optimizer, scheduler, scaler, feature_projectors,
        args.resume_optimizer, args.lr, args, loss_cfg, cfg_data,
        args.allow_loss_change_on_resume, args.allow_legacy_resume,
    )
    scheduler_world_size_report = None
    if args.resume_checkpoint:
        if args.resume_world_size is not None:
            scheduler_world_size_report = rescale_scheduler_for_world_size(
                scheduler,
                args.resume_world_size,
                world_size,
            )
        rebase_scheduler_after_resume(scheduler, args.epochs * steps_per_epoch)

    surface_mask = build_surface_mask(cfg_data, device)
    official15_indices = loss_cfg["official15_indices"]

    # Build weights — legacy mode applies official15 boost to all69 weights
    all69_weights, weight_summary = build_all69_weights(
        device, args.pressure_focus_levels, args.pressure_focus_weight,
        apply_official15_boost=loss_cfg["is_legacy"],
        official15_weight=loss_cfg["official15_weight"],
        official15_indices=official15_indices,
    )

    # Metadata
    metadata = build_full_metadata(args, loss_cfg, cfg, student_img_size, output_img_size,
                                    cfg_data, teacher_ckpt_sha256, init_report, seed)
    if scheduler_world_size_report is not None:
        metadata["scheduler_world_size_rescale"] = scheduler_world_size_report
    if fixed_subset_metadata is not None:
        metadata.update(fixed_subset_metadata)
    if main_process:
        (out_dir / "run_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"teacher_checkpoint_sha256={metadata['teacher_checkpoint_sha256']}")
        if scheduler_world_size_report is not None:
            print(
                "scheduler_world_size_rescale="
                + json.dumps(scheduler_world_size_report, ensure_ascii=False)
            )
        print(f"student_inherited_ratio={metadata['init_report_summary']['inherited_ratio']:.6f}")
        print(f"train_years={metadata['train_years']}")
        print(f"val_years={metadata['val_years']}")
        print(f"output_shape_contract={metadata['output_shape_contract']}")

    # DDP
    if world_size > 1:
        student = DDP(student, device_ids=[local_rank] if device.type == "cuda" else None,
                      output_device=local_rank if device.type == "cuda" else None,
                      find_unused_parameters=args.ddp_find_unused_parameters or args.feature_distill)
        if feature_projectors is not None:
            feature_projectors = DDP(feature_projectors,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
                find_unused_parameters=args.ddp_find_unused_parameters)

    if main_process:
        print(f"DDP world_size={world_size}, backend={args.dist_backend}")
        print(
            f"scheduler: steps_per_epoch={steps_per_epoch}, "
            f"total_steps={scheduler.total_steps}, warmup_steps={scheduler.warmup_steps}"
        )
        print_startup(args, cfg, loss_cfg, student_img_size, teacher_img_size,
                      output_img_size, device, amp_enabled, weight_summary, init_report, seed)
        print("official15 index mapping:")
        for row in official15_descriptions(official15_indices, getattr(cfg_data.dataset, "channels", None)):
            print(f"  {row}")

    if args.sanity_only:
        student.eval()
        data = next(iter(train_loader))
        if main_process:
            print_batch_structure(data)
        x, target = make_input_and_target(data, surface_mask, device)
        with torch.no_grad():
            with autocast_context(device, dtype, amp_enabled):
                teacher_pred = forward_69(teacher, x).float()
                student_pred, residual_base, residual_delta = forward_69(
                    student,
                    x,
                    student_img_size,
                    output_img_size,
                    residual_output=args.residual_output,
                    residual_scale=args.residual_scale,
                    return_parts=True,
                )
                student_pred = student_pred.float()
                validate_prediction_shape(student_pred, output_img_size)
                if student_pred.shape != teacher_pred.shape:
                    raise RuntimeError(
                        f"sanity shape mismatch: student={list(student_pred.shape)}, "
                        f"teacher={list(teacher_pred.shape)}"
                    )
                zero_feature = student_pred.new_zeros(())
                total_loss, components, gt_available = compute_decomposed_loss(
                    student_pred,
                    teacher_pred,
                    target,
                    all69_weights,
                    official15_indices,
                    loss_cfg,
                    zero_feature,
                )
        ensure_finite_loss(total_loss, "sanity-only")
        if main_process:
            print("SANITY CHECK COMPLETE - forward/loss/shape checks passed; no backward, step, or checkpoint save.")
            print(
                json.dumps(
                    {
                        "input_shape": list(x.shape),
                        "residual_output": args.residual_output,
                        "residual_base_shape": list(residual_base.shape) if residual_base is not None else None,
                        "residual_delta_shape": list(residual_delta.shape),
                        "student_shape": list(student_pred.shape),
                        "teacher_shape": list(teacher_pred.shape),
                        "target_shape": list(target.shape) if target is not None else None,
                        "metadata": residual_output_metadata(args),
                        "gt_available": gt_available,
                        "loss": components,
                    },
                    ensure_ascii=False,
                )
            )
        barrier()
        cleanup_distributed()
        return

    history_path = out_dir / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if main_process and history_path.exists() else []
    final_epoch = start_epoch + args.epochs
    printed_sanity = False
    printed_batch_structure = False
    consecutive_amp_overflows = 0

    for epoch in range(start_epoch, final_epoch):
        if fixed_train_sampler is None:
            set_sampler_epoch(_train_sampler, epoch)
            set_sampler_epoch(_val_sampler, epoch)
        student.train()
        if feature_projectors is not None:
            feature_projectors.train()
        train_rows = []
        pbar = tqdm(train_loader, desc=f"train {epoch+1}/{final_epoch}", disable=not main_process)
        for step, data in enumerate(pbar):
            if args.max_train_steps is not None and step >= args.max_train_steps:
                break
            if not printed_batch_structure:
                if main_process:
                    print_batch_structure(data)
                printed_batch_structure = True
            x, target = make_input_and_target(data, surface_mask, device)
            optimizer.zero_grad(set_to_none=True)
            if student_capture is not None:
                student_capture.clear()
            if teacher_capture is not None:
                teacher_capture.clear()

            with torch.no_grad():
                with autocast_context(device, dtype, amp_enabled):
                    teacher_pred = forward_69(teacher, x).float()
            with autocast_context(device, dtype, amp_enabled):
                student_pred = forward_69(
                    student,
                    x,
                    student_img_size,
                    output_img_size,
                    residual_output=args.residual_output,
                    residual_scale=args.residual_scale,
                ).float()
                validate_prediction_shape(student_pred, output_img_size)
                assert student_pred.shape == teacher_pred.shape

                loss_feature = student_pred.new_zeros(())
                if args.feature_distill and feature_projectors is not None:
                    loss_feature, _ = compute_feature_loss(
                        student_capture.features, teacher_capture.features,
                        feature_projectors, args.feature_stages,
                        student_channels, teacher_channels, not printed_sanity)

                total_loss, components, gt_available = compute_decomposed_loss(
                    student_pred, teacher_pred, target, all69_weights,
                    official15_indices, loss_cfg, loss_feature)

            ensure_finite_loss(total_loss, f"train epoch={epoch + 1} step={step + 1}")
            scaler.scale(total_loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.grad_clip)
            else:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=float("inf"))
            grad_norm_value = float(grad_norm.detach().cpu())
            grad_context = f"train epoch={epoch + 1} step={step + 1}"
            previous_overflows = consecutive_amp_overflows
            grad_state = classify_gradient_norm(
                grad_norm_value,
                scaler_enabled=scaler.is_enabled(),
                consecutive_overflows=previous_overflows,
                context=grad_context,
            )
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            amp_overflow = scaler.is_enabled() and amp_step_was_skipped(
                scale_before, scale_after
            )
            if amp_overflow:
                if grad_state["finite"]:
                    grad_state = classify_gradient_norm(
                        float("inf"),
                        scaler_enabled=True,
                        consecutive_overflows=previous_overflows,
                        context=grad_context,
                    )
                consecutive_amp_overflows = grad_state["consecutive_overflows"]
                if main_process:
                    print(
                        f"AMP overflow recovered at {grad_context}: "
                        f"grad_norm={grad_norm_value}, scale={scale_before}->{scale_after}, "
                        f"consecutive={consecutive_amp_overflows}",
                        flush=True,
                    )
            else:
                if grad_state["recoverable"]:
                    raise FloatingPointError(
                        f"non-finite gradient norm was not handled by GradScaler at "
                        f"{grad_context}: {grad_norm_value}"
                    )
                consecutive_amp_overflows = 0
            if not amp_overflow:
                scheduler.step()

            if not printed_sanity and main_process:
                print("first batch sanity:")
                print(f"  input={list(x.shape)}, student_out={list(student_pred.shape)}, teacher_out={list(teacher_pred.shape)}")
                print(f"  target={list(target.shape) if target is not None else None}, gt_available={gt_available}")
                for k, v in components.items():
                    print(f"  {k}={v:.6f}")
                printed_sanity = True

            components = dict(components)
            components["grad_norm"] = grad_norm_value if grad_state["finite"] else 0.0
            components["finite"] = 1.0 if grad_state["finite"] else 0.0
            components["amp_overflow"] = 1.0 if amp_overflow else 0.0
            components["amp_scale"] = scale_after
            train_rows.append(components)
            pbar.set_postfix({
                "total": f"{components['total']:.4f}",
                "t15": f"{components['teacher_15']:.4f}",
                "t_all": f"{components['teacher_all']:.4f}",
                "gt15": f"{components['gt_15']:.4f}",
            })

        # Validation
        student.eval()
        if feature_projectors is not None:
            feature_projectors.eval()
        val_rows = []
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"valid {epoch+1}/{final_epoch}", disable=not main_process)
            for step, data in enumerate(pbar):
                if args.max_val_steps is not None and step >= args.max_val_steps:
                    break
                x, target = make_input_and_target(data, surface_mask, device)
                if student_capture is not None:
                    student_capture.clear()
                if teacher_capture is not None:
                    teacher_capture.clear()
                with autocast_context(device, dtype, amp_enabled):
                    teacher_pred = forward_69(teacher, x).float()
                    student_pred = forward_69(
                        student,
                        x,
                        student_img_size,
                        output_img_size,
                        residual_output=args.residual_output,
                        residual_scale=args.residual_scale,
                    ).float()
                    validate_prediction_shape(student_pred, output_img_size)
                    assert student_pred.shape == teacher_pred.shape

                    loss_feature = student_pred.new_zeros(())
                    if args.feature_distill and feature_projectors is not None:
                        loss_feature, _ = compute_feature_loss(
                            student_capture.features, teacher_capture.features,
                            feature_projectors, args.feature_stages,
                            student_channels, teacher_channels, False)

                    total_loss, components, gt_available = compute_decomposed_loss(
                        student_pred, teacher_pred, target, all69_weights,
                        official15_indices, loss_cfg, loss_feature)
                    ensure_finite_loss(total_loss, f"valid epoch={epoch + 1} step={step + 1}")

                val_rows.append(components)
                pbar.set_postfix({
                    "teacher15_l1": f"{components['teacher_15']:.4f}",
                    "gt15_l1": f"{components['gt_15']:.4f}",
                    "gt_all_l1": f"{components['gt_all']:.4f}",
                })

        # Reduce training losses and the three explicitly internal validation L1 metrics.
        train_metric_payload = {
            k: avg(train_rows, k)
            for k in LOSS_KEYS + ["grad_norm", "finite", "amp_overflow", "amp_scale"]
        }
        train_metric_payload["amp_overflow_steps"] = sum(
            row.get("amp_overflow", 0.0) for row in train_rows
        )
        train_metrics = reduce_metrics(train_metric_payload, device)
        val_metrics = reduce_metrics(
            {
                "teacher15_l1": avg(val_rows, "teacher_15"),
                "gt15_l1": avg(val_rows, "gt_15"),
                "gt_all_l1": avg(val_rows, "gt_all"),
                "selection_total": avg(val_rows, "total"),
            },
            device,
        )

        val_t15 = val_metrics["teacher15_l1"]
        val_gt15 = val_metrics["gt15_l1"]
        val_total = val_metrics["selection_total"]

        log_row = {
            "epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
            "train_total": train_metrics["total"],
            "train_teacher_all": train_metrics["teacher_all"],
            "train_teacher_15": train_metrics["teacher_15"],
            "train_gt_15": train_metrics["gt_15"],
            "train_gt_all": train_metrics["gt_all"],
            "train_teacher15_l1": train_metrics["teacher_15"],
            "train_gt15_l1": train_metrics["gt_15"],
            "train_gt_all_l1": train_metrics["gt_all"],
            "train_feature": train_metrics["feature"],
            "train_grad_norm": train_metrics["grad_norm"],
            "train_grad_finite_fraction": train_metrics["finite"],
            "train_amp_overflow_steps": train_metrics["amp_overflow_steps"],
            "train_amp_overflow_fraction": train_metrics["amp_overflow"],
            "train_amp_scale": train_metrics["amp_scale"],
            "all_finite": bool(train_metrics["finite"] == 1.0),
            "val_total": val_total,
            "val_teacher15_l1": val_t15,
            "val_gt15_l1": val_gt15,
            "val_gt_all_l1": val_metrics["gt_all_l1"],
        }

        if main_process:
            history.append(log_row)
            history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
            print(log_row)

            improved = select_improved_checkpoints(
                val_total,
                val_t15,
                val_gt15,
                best_val_loss,
                best_teacher15,
                best_gt15,
            )

            # Save best teacher15
            if improved["teacher15"]:
                best_teacher15 = val_t15
                if not args.fixed_overfit_subset:
                    save_checkpoint(out_dir / "best_teacher15_loss.pth", student, optimizer, scheduler, scaler,
                                    epoch + 1, args, best_val_loss, best_teacher15, best_gt15,
                                    feature_projectors, metadata)
                save_param_only(out_dir / "best_teacher15_loss_param_only.pth", student, metadata, args.param_only_dtype)
                print(f"saved best_teacher15={best_teacher15:.6f}")

            # Save best gt15
            if improved["gt15"]:
                best_gt15 = val_gt15
                if not args.fixed_overfit_subset:
                    save_checkpoint(out_dir / "best_gt15_loss.pth", student, optimizer, scheduler, scaler,
                                    epoch + 1, args, best_val_loss, best_teacher15, best_gt15,
                                    feature_projectors, metadata)
                save_param_only(out_dir / "best_gt15_loss_param_only.pth", student, metadata, args.param_only_dtype)
                print(f"saved best_gt15={best_gt15:.6f}")

            # Save best internal validation objective. This is not an official W metric.
            if improved["val_total"]:
                best_val_loss = val_total
                if not args.fixed_overfit_subset:
                    save_checkpoint(out_dir / "best_val_loss.pth", student, optimizer, scheduler, scaler,
                                    epoch + 1, args, best_val_loss, best_teacher15, best_gt15,
                                    feature_projectors, metadata)
                print(f"saved best_val_loss={best_val_loss:.6f}")

            # Save last
            if should_save_full_checkpoint(
                args.fixed_overfit_subset, epoch + 1, final_epoch, args.save_every
            ):
                save_checkpoint(out_dir / "last.pth", student, optimizer, scheduler, scaler,
                                epoch + 1, args, best_val_loss, best_teacher15, best_gt15,
                                feature_projectors, metadata)
            if not args.fixed_overfit_subset and (epoch + 1) % args.save_every == 0:
                save_checkpoint(out_dir / f"epoch_{epoch+1}.pth", student, optimizer, scheduler, scaler,
                                epoch + 1, args, best_val_loss, best_teacher15, best_gt15,
                                feature_projectors, metadata)
        barrier()

    if student_capture is not None:
        student_capture.close()
    if teacher_capture is not None:
        teacher_capture.close()
    cleanup_distributed()

    if main_process and args.sanity_only:
        print("=" * 72)
        print("SANITY CHECK COMPLETE — no training performed.")
        print(f"Output dir: {out_dir}")
        print("=" * 72)


if __name__ == "__main__":
    main()

