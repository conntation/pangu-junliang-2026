#include <torch/extension.h>

#include "fused_bias_mask.h"

namespace {

constexpr int64_t kWindowTokens = 144;

void check_contiguous(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void validate(
    const at::Tensor& attn,
    const at::Tensor& bias,
    const at::Tensor& mask) {
  TORCH_CHECK(attn.is_cuda(), "attn must be on a ROCm/HIP device");
  TORCH_CHECK(attn.scalar_type() == at::kHalf, "attn must be float16");
  TORCH_CHECK(
      attn.dim() == 5,
      "attn must have shape [B, H, P, 144, 144]");
  TORCH_CHECK(
      attn.size(-1) == kWindowTokens && attn.size(-2) == kWindowTokens,
      "attn's final dimensions must be [144, 144]");
  check_contiguous(attn, "attn");

  TORCH_CHECK(bias.device() == attn.device(), "bias must be on attn's device");
  TORCH_CHECK(bias.scalar_type() == at::kHalf, "bias must be float16");
  TORCH_CHECK(
      bias.dim() == 4 && bias.size(0) == attn.size(1) &&
          bias.size(1) == attn.size(2) &&
          bias.size(2) == kWindowTokens && bias.size(3) == kWindowTokens,
      "bias must have shape [H, P, 144, 144]");
  check_contiguous(bias, "bias");

  TORCH_CHECK(mask.device() == attn.device(), "mask must be on attn's device");
  TORCH_CHECK(mask.scalar_type() == at::kChar, "mask must be int8");
  TORCH_CHECK(
      mask.dim() == 4 && mask.size(0) > 0 &&
          mask.size(1) == attn.size(2) &&
          mask.size(2) == kWindowTokens && mask.size(3) == kWindowTokens,
      "mask must have shape [W, P, 144, 144]");
  TORCH_CHECK(
      attn.size(0) % mask.size(0) == 0,
      "mask width W must divide attention batch B");
  check_contiguous(mask, "mask");
}

at::Tensor fused_bias_mask_add_binding(
    at::Tensor attn,
    const at::Tensor& bias,
    const at::Tensor& mask) {
  validate(attn, bias, mask);
  return launch_fused_bias_mask_add_hip(std::move(attn), bias, mask);
}

at::Tensor layer_norm_no_affine_binding(
    const at::Tensor& input,
    double epsilon) {
  TORCH_CHECK(input.is_cuda(), "input must be on a ROCm/HIP device");
  TORCH_CHECK(input.scalar_type() == at::kHalf, "input must be float16");
  TORCH_CHECK(input.dim() >= 2, "input must have at least two dimensions");
  TORCH_CHECK(
      input.size(-1) == 192 || input.size(-1) == 384 ||
          input.size(-1) == 768,
      "the optimized layer norm supports hidden sizes 192, 384, and 768");
  check_contiguous(input, "input");
  TORCH_CHECK(epsilon > 0.0, "epsilon must be positive");
  return launch_layer_norm_no_affine_hip(input, epsilon);
}

at::Tensor residual_add_layer_norm_binding(
    at::Tensor residual_destination,
    const at::Tensor& residual,
    double epsilon) {
  TORCH_CHECK(
      residual_destination.is_cuda(),
      "residual_destination must be on a ROCm/HIP device");
  TORCH_CHECK(
      residual_destination.scalar_type() == at::kHalf,
      "residual_destination must be float16");
  TORCH_CHECK(
      residual.device() == residual_destination.device(),
      "residual must be on residual_destination's device");
  TORCH_CHECK(residual.scalar_type() == at::kHalf, "residual must be float16");
  TORCH_CHECK(
      residual.sizes() == residual_destination.sizes(),
      "residual tensors must have identical shapes");
  TORCH_CHECK(
      residual_destination.dim() >= 2,
      "residual tensors must have at least two dimensions");
  TORCH_CHECK(
      residual_destination.size(-1) == 192 ||
          residual_destination.size(-1) == 384,
      "the optimized residual LayerNorm supports hidden sizes 192 and 384");
  check_contiguous(residual_destination, "residual_destination");
  check_contiguous(residual, "residual");
  TORCH_CHECK(epsilon > 0.0, "epsilon must be positive");
  return launch_residual_add_layer_norm_hip(
      std::move(residual_destination), residual, epsilon);
}

at::Tensor index_select_tokens_binding(
    const at::Tensor& input,
    const at::Tensor& index) {
  TORCH_CHECK(input.is_cuda(), "input must be on a ROCm/HIP device");
  TORCH_CHECK(input.scalar_type() == at::kHalf, "input must be float16");
  TORCH_CHECK(input.dim() == 3, "input must have shape [B, N, C]");
  TORCH_CHECK(
      input.size(2) == 192 || input.size(2) == 384,
      "the optimized token gather supports 192/384 channels");
  TORCH_CHECK(index.device() == input.device(), "index must be on input's device");
  TORCH_CHECK(index.scalar_type() == at::kInt, "index must be int32");
  TORCH_CHECK(index.dim() == 1, "index must be one-dimensional");
  check_contiguous(input, "input");
  check_contiguous(index, "index");
  return launch_index_select_tokens_hip(input, index);
}

at::Tensor layer_norm_gather_binding(
    const at::Tensor& input,
    const at::Tensor& index,
    double epsilon) {
  TORCH_CHECK(input.is_cuda(), "input must be on a ROCm/HIP device");
  TORCH_CHECK(input.scalar_type() == at::kHalf, "input must be float16");
  TORCH_CHECK(input.dim() == 3, "input must have shape [B, N, C]");
  TORCH_CHECK(
      input.size(2) == 192 || input.size(2) == 384,
      "the fused LayerNorm gather supports 192/384 channels");
  TORCH_CHECK(index.device() == input.device(), "index must be on input's device");
  TORCH_CHECK(index.scalar_type() == at::kInt, "index must be int32");
  TORCH_CHECK(index.dim() == 1, "index must be one-dimensional");
  check_contiguous(input, "input");
  check_contiguous(index, "index");
  TORCH_CHECK(epsilon > 0.0, "epsilon must be positive");
  return launch_layer_norm_gather_hip(input, index, epsilon);
}

at::Tensor affine_fp16_to_fp32_binding(
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& std,
    at::Tensor output) {
  TORCH_CHECK(input.is_cuda(), "input must be on a ROCm/HIP device");
  TORCH_CHECK(input.scalar_type() == at::kHalf, "input must be float16");
  TORCH_CHECK(input.dim() == 4, "input must have shape [B,C,H,W]");
  TORCH_CHECK(input.size(3) % 2 == 0, "input width must be even");
  TORCH_CHECK(input.stride(3) == 1, "input width must be contiguous");

  TORCH_CHECK(mean.device() == input.device(), "mean must be on input's device");
  TORCH_CHECK(mean.scalar_type() == at::kFloat, "mean must be float32");
  TORCH_CHECK(std.device() == input.device(), "std must be on input's device");
  TORCH_CHECK(std.scalar_type() == at::kFloat, "std must be float32");
  TORCH_CHECK(
      mean.dim() == 4 && std.dim() == 4 && mean.size(0) == 1 &&
          std.size(0) == 1 && mean.size(1) == input.size(1) &&
          std.size(1) == input.size(1) && mean.size(2) == 1 &&
          std.size(2) == 1 && mean.size(3) == 1 && std.size(3) == 1,
      "channel affine tensors must have shape [1,C,1,1]");
  check_contiguous(mean, "mean");
  check_contiguous(std, "std");

  TORCH_CHECK(output.device() == input.device(), "output must be on input's device");
  TORCH_CHECK(output.scalar_type() == at::kFloat, "output must be float32");
  TORCH_CHECK(output.sizes() == input.sizes(), "output must match input shape");
  check_contiguous(output, "output");
  return launch_affine_fp16_to_fp32_hip(input, mean, std, std::move(output));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "fused_bias_mask_add_",
      &fused_bias_mask_add_binding,
      pybind11::arg("attn"),
      pybind11::arg("bias"),
      pybind11::arg("mask"),
      R"doc(
In-place gfx936 FP16 bias plus INT8 0/-100 mask addition.

The two additions use separate native FP16 additions, preserving the source
operation's rounding point before the mask is applied.  Softmax is deliberately
left to PyTorch.
)doc");
  module.def(
      "layer_norm_no_affine",
      &layer_norm_no_affine_binding,
      pybind11::arg("input"),
      pybind11::arg("epsilon"),
      R"doc(
Out-of-place gfx936 FP16 LayerNorm for hidden sizes 192/384/768 without affine.
The reduction and normalization arithmetic use FP32.
)doc");
  module.def(
      "residual_add_layer_norm_",
      &residual_add_layer_norm_binding,
      pybind11::arg("residual_destination"),
      pybind11::arg("residual"),
      pybind11::arg("epsilon"),
      R"doc(
Add an FP16 residual in place, then return its no-affine LayerNorm result.
The addition retains its FP16 rounding point; normalization uses FP32.
)doc");
  module.def(
      "index_select_tokens",
      &index_select_tokens_binding,
      pybind11::arg("input"),
      pybind11::arg("index"),
      R"doc(
gfx936 token-axis gather for contiguous FP16 [B,N,192/384] tensors.
An int32 index of -1 produces a zero token, fusing padding fill.
)doc");
  module.def(
      "layer_norm_gather",
      &layer_norm_gather_binding,
      pybind11::arg("input"),
      pybind11::arg("index"),
      pybind11::arg("epsilon"),
      R"doc(Fuse no-affine LayerNorm with token gather and -1 padding.)doc");
  module.def(
      "affine_fp16_to_fp32",
      &affine_fp16_to_fp32_binding,
      pybind11::arg("input"),
      pybind11::arg("mean"),
      pybind11::arg("std"),
      pybind11::arg("output"),
      R"doc(Convert strided NCHW FP16 input to FP32 and apply channel affine.)doc");
}

