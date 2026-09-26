
#!/usr/bin/env python3
"""Quantize Earth-position-bias tables for compact Pangu checkpoints.

The existing W8A16 checkpoint is dominated by sixteen fp16
``earth_position_bias_table`` tensors. This utility replaces each table with
blockwise symmetric INT8 or packed INT4 storage. The matching runtime loader is
implemented in ``inference.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


EPB_NAME = "earth_position_bias_table"
FORMAT_INT8 = "pangu-epb-int8-v1"
FORMAT_INT4 = "pangu-epb-int4-v1"
FORMAT_ZERO = "pangu-epb-zero-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", required=True)
    parser.add_argument("--checkpoint-out", required=True)
    parser.add_argument(
        "--bits",
        type=int,
        choices=(0, 4, 8),
        default=8,
        help="Storage precision. 0 removes EPB values for an ablation.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Rows per scale block. Defaults: INT8=72, INT4=18.",
    )
    return parser.parse_args()


def load_checkpoint(path: str | Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must be a dict, got {type(checkpoint)!r}")
    return checkpoint


def unwrap_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("model_state_dict must be a dict")
    return state


def quantize_blockwise(
    value: torch.Tensor,
    bits: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim != 3:
        raise ValueError(f"expected a 3D EPB table, got shape={tuple(value.shape)}")
    rows = value.shape[0]
    if rows % block_size != 0:
        raise ValueError(f"EPB rows={rows} is not divisible by block_size={block_size}")

    qmax = 127 if bits == 8 else 7
    blocks = value.detach().float().reshape(
        rows // block_size, block_size, value.shape[1], value.shape[2]
    )
    scale = (
        blocks.abs().amax(dim=1).clamp_min(1e-12) / float(qmax)
    ).half().contiguous()
    quantized = (
        torch.round(blocks / scale.float().unsqueeze(1))
        .clamp(-qmax, qmax)
        .to(torch.int8)
        .reshape_as(value)
        .contiguous()
    )
    return quantized, scale


def pack_int4(quantized: torch.Tensor) -> torch.Tensor:
    if quantized.shape[0] % 2 != 0:
        raise ValueError("INT4 packing requires an even first dimension")
    encoded = (quantized.to(torch.int16) + 8).to(torch.uint8)
    low = encoded[0::2]
    high = encoded[1::2]
    return (low | (high << 4)).contiguous()


def main() -> None:
    args = parse_args()
    block_size = args.block_size or (72 if args.bits == 8 else 18)
    checkpoint = load_checkpoint(args.checkpoint_in)
    state = unwrap_state(checkpoint)

    output_state: dict[str, torch.Tensor] = {}
    original_bytes = 0
    quantized_bytes = 0
    quantized_names: list[str] = []

    for name, value in state.items():
        if not (torch.is_tensor(value) and name.endswith(EPB_NAME)):
            output_state[name] = value
            continue

        prefix = name[: -len(EPB_NAME)]
        original_bytes += value.numel() * value.element_size()
        if args.bits == 0:
            marker = torch.zeros((), dtype=torch.uint8)
            output_state[f"{prefix}earth_position_bias_zero"] = marker
            quantized_bytes += marker.numel() * marker.element_size()
            quantized_names.append(name)
            continue

        quantized, scale = quantize_blockwise(value, args.bits, block_size)
        if args.bits == 8:
            output_state[f"{prefix}earth_position_bias_qweight"] = quantized
            quantized_bytes += quantized.numel() * quantized.element_size()
        else:
            packed = pack_int4(quantized)
            output_state[f"{prefix}earth_position_bias_qweight_packed"] = packed
            quantized_bytes += packed.numel() * packed.element_size()
        output_state[f"{prefix}earth_position_bias_scale"] = scale
        quantized_bytes += scale.numel() * scale.element_size()
        quantized_names.append(name)

    if not quantized_names:
        raise RuntimeError(f"no {EPB_NAME!r} tensors found in {args.checkpoint_in}")

    metadata = {
        "format": (
            FORMAT_ZERO
            if args.bits == 0
            else FORMAT_INT8
            if args.bits == 8
            else FORMAT_INT4
        ),
        "bits": args.bits,
        "scheme": "symmetric_blockwise_dim0",
        "block_size": block_size,
        "scale_dtype": "fp16",
        "quantized_tensors": quantized_names,
        "original_bytes": original_bytes,
        "quantized_bytes": quantized_bytes,
    }
    output_checkpoint = dict(checkpoint)
    output_checkpoint["model_state_dict"] = output_state
    output_checkpoint["epb_quant"] = metadata

    output_path = Path(args.checkpoint_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, output_path)
    print(f"saved: {output_path}")
    print(f"format: {metadata['format']}, block_size={block_size}")
    print(f"quantized EPB tensors: {len(quantized_names)}")
    print(f"EPB bytes: {original_bytes} -> {quantized_bytes}")
    print(f"file size: {Path(args.checkpoint_in).stat().st_size} -> {output_path.stat().st_size}")


if __name__ == "__main__":
    main()

