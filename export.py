import torch
from pathlib import Path

SRC = Path("data/checkpoints/model_infer_bf16.pth")
DST = Path("data/checkpoints/model_slim_bf16.pth")

DROP_KEY_PATTERNS = (
    ".attn_mask",
    ".earth_position_index",
    "device_buffer",
    ".Fuser."
)

def should_drop(key: str) -> bool:
    return any(pattern in key for pattern in DROP_KEY_PATTERNS)

ckpt = torch.load(SRC, map_location="cpu", weights_only=False)
state = ckpt["model_state_dict"]

slim_state = {
    key: value
    for key, value in state.items()
    if not should_drop(key)
}

torch.save({"model_state_dict": slim_state}, DST)

print(f"old: {SRC.stat().st_size / 1024**2:.2f} MiB")
print(f"new: {DST.stat().st_size / 1024**2:.2f} MiB")
print(f"dropped: {len(state) - len(slim_state)}")
