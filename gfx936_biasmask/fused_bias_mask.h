#pragma once

#include <ATen/ATen.h>

at::Tensor launch_fused_bias_mask_add_hip(
    at::Tensor attn,
    const at::Tensor& bias,
    const at::Tensor& mask);

at::Tensor launch_layer_norm_no_affine_hip(
    const at::Tensor& input,
    double epsilon);

at::Tensor launch_residual_add_layer_norm_hip(
    at::Tensor residual_destination,
    const at::Tensor& residual,
    double epsilon);

at::Tensor launch_index_select_tokens_hip(
    const at::Tensor& input,
    const at::Tensor& index);

at::Tensor launch_layer_norm_gather_hip(
    const at::Tensor& input,
    const at::Tensor& index,
    double epsilon);

at::Tensor launch_affine_fp16_to_fp32_hip(
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& std,
    at::Tensor output);

