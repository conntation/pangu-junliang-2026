#!/usr/bin/env python3
"""Pack selected W8A16 Linear weights as blockwise W4 on disk.

This is a storage optimization. At model load, inference.py unpacks the values
once to INT8 and W8A16-style fp16 compute continues. Only selected 2D Linear
weights are transformed; convolution and recovery weights remain W8.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", required=True)
    parser.add_argument("--checkpoint-out", required=True)
    parser.add_argument(
        "--mode",
        choices=("layer2", "layer3", "deep"),
        default="deep",
    )
    parser.add_argument("--bits", type=int, choices=(4, 5), default=4)
    parser.add_argument("--block-size", type=int, default=16)
    return parser.parse_args()


def selected(name: str, mode: str) -> bool:
    if mode == "layer2":
        return name.startswith("layer2.")
    if mode == "layer3":
        return name.startswith("layer3.")
    return name.startswith("layer2.") or name.startswith("layer3.")


def pack_last_dim(quantized: torch.Tensor) -> torch.Tensor:
    if quantized.shape[-1] % 2 != 0:
        raise ValueError(f"last dimension must be even: {tuple(quantized.shape)}")
    encoded = (quantized.to(torch.int16) + 8).to(torch.uint8)
    return (encoded[..., 0::2] | (encoded[..., 1::2] << 4)).contiguous()


def pack_5bit_last_dim(quantized: torch.Tensor) -> torch.Tensor:
    if quantized.shape[-1] % 8 != 0:
        raise ValueError(
            f"last dimension must be divisible by 8: {tuple(quantized.shape)}"
        )
    encoded = (quantized.to(torch.int64) + 16).reshape(
        *quantized.shape[:-1], -1, 8
    )
    word = torch.zeros(encoded.shape[:-1], dtype=torch.int64)
    for index in range(8):
        word |= encoded[..., index] << (5 * index)
    packed = torch.stack(
        [((word >> (8 * index)) & 0xFF).to(torch.uint8) for index in range(5)],
        dim=-1,
    )
    return packed.reshape(*quantized.shape[:-1], -1).contiguous()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint_in,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint["model_state_dict"]
    output_state = dict(state)
    converted: list[str] = []
    original_bytes = 0
    packed_bytes = 0

    for name, qweight in state.items():
        if not (
            name.endswith(".qweight")
            and torch.is_tensor(qweight)
            and qweight.ndim == 2
            and selected(name, args.mode)
        ):
            continue

        prefix = name[: -len(".qweight")]
        scale_name = f"{prefix}.scale"
        if scale_name not in state:
            raise KeyError(f"missing W8 scale: {scale_name}")
        old_scale = state[scale_name].float()
        weight = qweight.float() * old_scale[:, None]
        out_features, in_features = weight.shape
        if in_features % args.block_size != 0:
            raise ValueError(
                f"{name}: in_features={in_features} is not divisible by "
                f"block_size={args.block_size}"
            )

        blocks = weight.reshape(
            out_features,
            in_features // args.block_size,
            args.block_size,
        )
        qmax = 2 ** (args.bits - 1) - 1
        new_scale = (
            blocks.abs().amax(dim=2).clamp_min(1e-12) / float(qmax)
        ).half().contiguous()
        quantized = (
            torch.round(blocks / new_scale.float().unsqueeze(2))
            .clamp(-qmax, qmax)
            .to(torch.int8)
            .reshape_as(qweight)
        )
        packed = (
            pack_last_dim(quantized)
            if args.bits == 4
            else pack_5bit_last_dim(quantized)
        )

        del output_state[name]
        packed_suffix = "qweight_packed" if args.bits == 4 else "qweight_packed5"
        output_state[f"{prefix}.{packed_suffix}"] = packed
        output_state[scale_name] = new_scale
        original_bytes += (
            qweight.numel() * qweight.element_size()
            + state[scale_name].numel() * state[scale_name].element_size()
        )
        packed_bytes += (
            packed.numel() * packed.element_size()
            + new_scale.numel() * new_scale.element_size()
        )
        converted.append(prefix)

    if not converted:
        raise RuntimeError("no matching W8 Linear tensors were found")

    output_checkpoint = dict(checkpoint)
    output_checkpoint["model_state_dict"] = output_state
    output_checkpoint["w4_storage"] = {
        "format": f"pangu-w{args.bits}a16-blockwise-v1",
        "bits": args.bits,
        "mode": args.mode,
        "block_size": args.block_size,
        "scheme": "symmetric_per_output_input_block",
        "packed_dim": -1,
        "converted_modules": converted,
        "original_bytes": original_bytes,
        "packed_bytes": packed_bytes,
    }
    output_path = Path(args.checkpoint_out)
    torch.save(output_checkpoint, output_path)
    print(f"saved: {output_path}")
    print(f"converted Linear modules: {len(converted)}")
    print(f"selected weight bytes: {original_bytes} -> {packed_bytes}")
    print(
        f"file size: {Path(args.checkpoint_in).stat().st_size} "
        f"-> {output_path.stat().st_size}"
    )


if __name__ == "__main__":
    main()

