import torch

def get_state_dict(ckpt):
    """
    兼容常见 checkpoint 格式：
    1. 直接是 state_dict
    2. {'model': state_dict}
    3. {'state_dict': state_dict}
    4. {'model_state_dict': state_dict}
    """
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "model_state_dict", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt

def convert_state_dict_to_fp16(state_dict):
    new_state_dict = {}

    for k, v in state_dict.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            new_state_dict[k] = v.half()
        else:
            new_state_dict[k] = v

    return new_state_dict

src_path = "./data/student_3/best_gt15_loss_param_only.pth"          # 你的原始 FP32 推理权重
dst_path = "./data/student_3/bf16.pth"     # 输出 FP16 推理权重

ckpt = torch.load(src_path, map_location="cpu")
state_dict = get_state_dict(ckpt)

fp16_state_dict = convert_state_dict_to_fp16(state_dict)

torch.save({"model_state_dict": fp16_state_dict}, dst_path)

print(f"saved fp16 checkpoint to {dst_path}")

