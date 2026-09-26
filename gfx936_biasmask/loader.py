"""Lazy loader for the accepted gfx936 Pangu inference kernels."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


SOURCE_DIR = Path(__file__).resolve().parent
EXTENSION_NAME = "gfx936_bias_mask_add_ext"
TARGET_ARCH = "gfx936"
_EXTENSION: ModuleType | None = None


def _detected_arch() -> str:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]


def load_extension(*, verbose: bool = False) -> ModuleType:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("a ROCm PyTorch runtime and visible HIP device are required")
    detected = _detected_arch()
    if detected != TARGET_ARCH:
        raise RuntimeError(f"expected {TARGET_ARCH}, detected {detected or 'unknown'}")
    _EXTENSION = load(
        name=EXTENSION_NAME,
        sources=[
            str(SOURCE_DIR / "binding.cpp"),
            str(SOURCE_DIR / "fused_bias_mask_kernel.hip"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", f"--offload-arch={TARGET_ARCH}"],
        extra_include_paths=[str(SOURCE_DIR)],
        with_cuda=True,
        verbose=verbose,
    )
    return _EXTENSION


def fused_bias_mask_add_(
    attn: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return load_extension().fused_bias_mask_add_(attn, bias, mask)


def layer_norm_no_affine(
    input: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    return load_extension().layer_norm_no_affine(input, epsilon)


def residual_add_layer_norm_(
    residual_destination: torch.Tensor,
    residual: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    return load_extension().residual_add_layer_norm_(
        residual_destination,
        residual,
        epsilon,
    )


def index_select_tokens(
    input: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    return load_extension().index_select_tokens(input, index)


def layer_norm_gather(
    input: torch.Tensor,
    index: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    return load_extension().layer_norm_gather(input, index, epsilon)


def affine_fp16_to_fp32(
    input: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Apply channel-wise FP32 affine conversion into a caller-owned buffer."""
    return load_extension().affine_fp16_to_fp32(input, mean, std, output)

