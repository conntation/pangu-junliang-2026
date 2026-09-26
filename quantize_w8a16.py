#!/usr/bin/env python3
"""W8A16 weight-only quantization utilities for the Pangu inference model.

The exported checkpoint stores selected weights as int8 plus per-output-channel
scales. Activations stay in fp16 during inference. To run the quantized weights
directly, replace supported modules with the W8A16 wrappers before loading the
quantized state dict:

    from quantize_w8a16 import load_quantized_w8a16_state_dict, replace_modules_with_w8a16

    model = build_model(...)
    replace_modules_with_w8a16(model)
    state = load_quantized_w8a16_state_dict("data/checkpoints/model_bak_w8a16.pth")
    model.load_state_dict(state, strict=False)

If you only need a compatibility check, use --mode dequantize to export an fp16
checkpoint that existing inference scripts can load without module replacement.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _reverse_repeat_tuple


QUANT_FORMAT = "pangu-w8a16-v1"
DEFAULT_EXCLUDES = (
    "*.norm.*",
    "*norm*.weight",
    "*.bias",
    "*.attn_mask",
    "*.earth_position_index",
    "*device_buffer*",
)
DEFAULT_DROP_PATTERNS = (
    "Fuser.*",
    "*.Fuser.*",
    "Sampler.*",
    "*.Sampler.*",
    "Reconvery.*",
    "*.Reconvery.*",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint-in", default="data/checkpoints/model_bak.pth")
    parser.add_argument("--checkpoint-out", default="data/checkpoints/model_bak_w8a16.pth")
    parser.add_argument(
        "--mode",
        choices=("quantize", "dequantize"),
        default="quantize",
        help="quantize fp checkpoint to W8A16, or dequantize W8A16 checkpoint back to fp16",
    )
    parser.add_argument(
        "--scale-dtype",
        choices=("fp16", "fp32"),
        default="fp16",
        help="dtype used to store quantization scales",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="fnmatch pattern to skip; may be repeated",
    )
    parser.add_argument(
        "--min-numel",
        type=int,
        default=1024,
        help="skip tiny tensors where int8 metadata is not worth it",
    )
    return parser.parse_args()


def unwrap_checkpoint(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise TypeError(f"unsupported checkpoint type: {type(checkpoint)!r}")

    if not isinstance(state, dict):
        raise TypeError(f"unsupported model_state_dict type: {type(state)!r}")
    return state


def load_checkpoint(path: str | Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def scale_dtype_from_name(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.float32


def matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def should_quantize_tensor(
    name: str,
    value: torch.Tensor,
    excludes: Iterable[str],
    min_numel: int,
) -> bool:
    return (
        name.endswith(".weight")
        and value.is_floating_point()
        and value.ndim >= 2
        and value.numel() >= min_numel
        and not matches_any(name, excludes)
    )


def quantize_per_channel_symmetric(
    weight: torch.Tensor,
    scale_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight over dim 0 with symmetric int8 ranges."""
    cpu_weight = weight.detach().float().cpu()
    flat = cpu_weight.reshape(cpu_weight.shape[0], -1)
    max_abs = flat.abs().amax(dim=1)
    scale = torch.where(max_abs > 0, max_abs / 127.0, torch.ones_like(max_abs))
    qweight = torch.round(flat / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return qweight.reshape_as(cpu_weight).contiguous(), scale.to(scale_dtype).contiguous()


def dequantize_per_channel_symmetric(
    qweight: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.float16,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is not None:
        qweight = qweight.to(device=device)
        scale = scale.to(device=device)
    shape = [1] * qweight.ndim
    shape[0] = -1
    return qweight.to(dtype) * scale.to(dtype).reshape(shape)


def quantize_state_dict_w8a16(
    state: dict[str, torch.Tensor],
    scale_dtype: torch.dtype = torch.float16,
    excludes: Iterable[str] = DEFAULT_EXCLUDES,
    min_numel: int = 1024,
    drop_patterns: Iterable[str] = DEFAULT_DROP_PATTERNS,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    new_state: dict[str, torch.Tensor] = {}
    quantized_keys: list[str] = []
    skipped_keys: list[str] = []
    original_bytes = 0
    quantized_bytes = 0

    for name, value in state.items():
        if torch.is_tensor(value):
            original_bytes += value.numel() * value.element_size()

        if matches_any(name, drop_patterns):
            skipped_keys.append(name)
            continue

        if torch.is_tensor(value) and should_quantize_tensor(name, value, excludes, min_numel):
            prefix = name[: -len(".weight")]
            qweight, scale = quantize_per_channel_symmetric(value, scale_dtype=scale_dtype)
            new_state[f"{prefix}.qweight"] = qweight
            new_state[f"{prefix}.scale"] = scale
            quantized_keys.append(prefix)
            quantized_bytes += qweight.numel() * qweight.element_size()
            quantized_bytes += scale.numel() * scale.element_size()
            continue

        if torch.is_tensor(value) and value.is_floating_point():
            new_state[name] = value.detach().cpu().half()
        else:
            new_state[name] = value
        if torch.is_tensor(new_state[name]):
            quantized_bytes += new_state[name].numel() * new_state[name].element_size()

    metadata: dict[str, object] = {
        "format": QUANT_FORMAT,
        "scheme": "symmetric_per_channel_dim0",
        "activation_dtype": "fp16",
        "weight_dtype": "int8",
        "scale_dtype": "fp16" if scale_dtype == torch.float16 else "fp32",
        "quantized_keys": quantized_keys,
        "skipped_keys": skipped_keys,
        "original_bytes": original_bytes,
        "quantized_bytes": quantized_bytes,
    }
    return new_state, metadata


def dequantize_state_dict_w8a16(
    state: dict[str, torch.Tensor],
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    new_state: dict[str, torch.Tensor] = {}
    consumed_scales: set[str] = set()

    for name, value in state.items():
        if not name.endswith(".qweight"):
            continue
        prefix = name[: -len(".qweight")]
        scale_name = f"{prefix}.scale"
        if scale_name not in state:
            raise KeyError(f"missing scale tensor for {name}: expected {scale_name}")
        new_state[f"{prefix}.weight"] = dequantize_per_channel_symmetric(
            value,
            state[scale_name],
            dtype=dtype,
        )
        consumed_scales.add(scale_name)

    for name, value in state.items():
        if name.endswith(".qweight") or name in consumed_scales:
            continue
        if torch.is_tensor(value) and value.is_floating_point():
            new_state[name] = value.detach().cpu().to(dtype)
        else:
            new_state[name] = value
    return new_state


def load_quantized_w8a16_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    return unwrap_checkpoint(load_checkpoint(path))


def load_dequantized_w8a16_state_dict(
    path: str | Path,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    return dequantize_state_dict_w8a16(load_quantized_w8a16_state_dict(path), dtype=dtype)


class W8A16Linear(nn.Module):
    def __init__(self, qweight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.in_features = qweight.shape[1]
        self.out_features = qweight.shape[0]
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scale", scale.contiguous())
        self.register_buffer("bias", None if bias is None else bias.detach().contiguous())

    @classmethod
    def from_float(cls, module: nn.Linear) -> "W8A16Linear":
        qweight, scale = quantize_per_channel_symmetric(module.weight, scale_dtype=torch.float16)
        bias = None if module.bias is None else module.bias.detach().half()
        return cls(qweight, scale, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype if x.is_floating_point() else torch.float16
        weight = dequantize_per_channel_symmetric(
            self.qweight,
            self.scale,
            dtype=dtype,
            device=x.device,
        )
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=dtype)
        return F.linear(x, weight, bias)


class W8A16Conv3d(nn.Module):
    def __init__(self, module: nn.Conv3d):
        super().__init__()
        qweight, scale = quantize_per_channel_symmetric(module.weight, scale_dtype=torch.float16)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", None if module.bias is None else module.bias.detach().half())
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups
        self.padding_mode = module.padding_mode
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(module.padding, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype if x.is_floating_point() else torch.float16
        weight = dequantize_per_channel_symmetric(
            self.qweight,
            self.scale,
            dtype=dtype,
            device=x.device,
        )
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=dtype)
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = 0
        else:
            padding = self.padding
        return F.conv3d(x, weight, bias, self.stride, padding, self.dilation, self.groups)


class W8A16ConvTranspose3d(nn.Module):
    def __init__(self, module: nn.ConvTranspose3d):
        super().__init__()
        qweight, scale = quantize_per_channel_symmetric(module.weight, scale_dtype=torch.float16)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", None if module.bias is None else module.bias.detach().half())
        self.stride = module.stride
        self.padding = module.padding
        self.output_padding = module.output_padding
        self.groups = module.groups
        self.dilation = module.dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype if x.is_floating_point() else torch.float16
        weight = dequantize_per_channel_symmetric(
            self.qweight,
            self.scale,
            dtype=dtype,
            device=x.device,
        )
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=dtype)
        return F.conv_transpose3d(
            x,
            weight,
            bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )


def replace_modules_with_w8a16(
    module: nn.Module,
    skip_name_patterns: Iterable[str] = (),
    prefix: str = "",
) -> nn.Module:
    """In-place replacement of supported modules with W8A16 wrappers."""
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if matches_any(full_name, skip_name_patterns):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, child_name, W8A16Linear.from_float(child))
        elif isinstance(child, nn.Conv3d):
            setattr(module, child_name, W8A16Conv3d(child))
        elif isinstance(child, nn.ConvTranspose3d):
            setattr(module, child_name, W8A16ConvTranspose3d(child))
        else:
            replace_modules_with_w8a16(child, skip_name_patterns, prefix=full_name)
    return module


def save_checkpoint(state: dict[str, torch.Tensor], metadata: dict[str, object], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": state, "w8a16": metadata}, output_path)


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint_in)
    state = unwrap_checkpoint(checkpoint)

    if args.mode == "quantize":
        excludes = tuple(DEFAULT_EXCLUDES) + tuple(args.exclude)
        quant_state, metadata = quantize_state_dict_w8a16(
            state,
            scale_dtype=scale_dtype_from_name(args.scale_dtype),
            excludes=excludes,
            min_numel=args.min_numel,
        )
        save_checkpoint(quant_state, metadata, args.checkpoint_out)
        old_mb = os.path.getsize(args.checkpoint_in) / 1024 / 1024
        new_mb = os.path.getsize(args.checkpoint_out) / 1024 / 1024
        print(f"saved W8A16 checkpoint: {args.checkpoint_out}")
        print(f"quantized tensors: {len(metadata['quantized_keys'])}")
        print(f"skipped tensors: {len(metadata['skipped_keys'])}")
        print(f"file size: {old_mb:.2f} MB -> {new_mb:.2f} MB")
        return

    dequant_state = dequantize_state_dict_w8a16(state, dtype=torch.float16)
    save_checkpoint(
        dequant_state,
        {
            "format": f"{QUANT_FORMAT}-dequantized-fp16",
            "source": str(args.checkpoint_in),
        },
        args.checkpoint_out,
    )
    print(f"saved dequantized fp16 checkpoint: {args.checkpoint_out}")


if __name__ == "__main__":
    main()

