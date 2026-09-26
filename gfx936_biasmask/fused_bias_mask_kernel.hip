#include <ATen/ATen.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <limits>

#include "fused_bias_mask.h"

namespace {

constexpr int kWindowTokens = 144;
constexpr int kElementsPerPlane = kWindowTokens * kWindowTokens;
constexpr int kPairsPerPlane = kElementsPerPlane / 2;
constexpr int kThreadsPerBlock = 256;
constexpr int kWavefrontSize = 64;
constexpr int kRowsPerSoftmaxBlock = kThreadsPerBlock / kWavefrontSize;

__global__ __launch_bounds__(kThreadsPerBlock) void fused_bias_mask_add_kernel(
    __half* __restrict__ attn,
    const __half* __restrict__ bias,
    const int8_t* __restrict__ mask,
    int heads,
    int pressure,
    int mask_widths) {
  int64_t plane = static_cast<int64_t>(blockIdx.x);
  const int p = static_cast<int>(plane % pressure);
  plane /= pressure;
  const int h = static_cast<int>(plane % heads);
  const int64_t b = plane / heads;
  const int mask_width = static_cast<int>(b % mask_widths);

  const int64_t attn_pair_base =
      ((b * heads + h) * pressure + p) * kPairsPerPlane;
  const int64_t bias_pair_base =
      (static_cast<int64_t>(h) * pressure + p) * kPairsPerPlane;
  const int64_t mask_base =
      (static_cast<int64_t>(mask_width) * pressure + p) * kElementsPerPlane;

  auto* attn_pairs = reinterpret_cast<__half2*>(attn);
  const auto* bias_pairs = reinterpret_cast<const __half2*>(bias);
  for (int pair = static_cast<int>(threadIdx.x); pair < kPairsPerPlane;
       pair += kThreadsPerBlock) {
    const __half2 after_bias = __hadd2(
        attn_pairs[attn_pair_base + pair],
        bias_pairs[bias_pair_base + pair]);
    const int element = pair * 2;
    // The persistent mask contract is exactly {0, -100}.  Selecting the two
    // exactly representable FP16 constants avoids an integer-to-float
    // conversion for every attention element.
    const __half mask0 = mask[mask_base + element] == 0
        ? __float2half_rn(0.0f)
        : __float2half_rn(-100.0f);
    const __half mask1 = mask[mask_base + element + 1] == 0
        ? __float2half_rn(0.0f)
        : __float2half_rn(-100.0f);
    const __half2 mask_pair = __halves2half2(mask0, mask1);
    attn_pairs[attn_pair_base + pair] = __hadd2(after_bias, mask_pair);
  }
}

__device__ __forceinline__ float wave_reduce_sum(float value) {
  value += __shfl_down(value, 32, kWavefrontSize);
  value += __shfl_down(value, 16, kWavefrontSize);
  value += __shfl_down(value, 8, kWavefrontSize);
  value += __shfl_down(value, 4, kWavefrontSize);
  value += __shfl_down(value, 2, kWavefrontSize);
  value += __shfl_down(value, 1, kWavefrontSize);
  return __shfl(value, 0, kWavefrontSize);
}

template <int Hidden>
__global__ __launch_bounds__(kThreadsPerBlock) void layer_norm_no_affine_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int64_t rows,
    float epsilon) {
  constexpr int kValuesPerLane = Hidden / kWavefrontSize;
  const int lane = static_cast<int>(threadIdx.x) & (kWavefrontSize - 1);
  const int wave = static_cast<int>(threadIdx.x) / kWavefrontSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kRowsPerSoftmaxBlock + wave;
  if (row >= rows) {
    return;
  }

  const int64_t base = row * Hidden;
  float values[kValuesPerLane];
  float local_sum = 0.0f;
  float local_square_sum = 0.0f;
#pragma unroll
  for (int item = 0; item < kValuesPerLane; ++item) {
    const int column = lane + item * kWavefrontSize;
    const float value = __half2float(input[base + column]);
    values[item] = value;
    local_sum += value;
    local_square_sum = fmaf(value, value, local_square_sum);
  }

  const float mean = wave_reduce_sum(local_sum) * (1.0f / Hidden);
  const float square_mean =
      wave_reduce_sum(local_square_sum) * (1.0f / Hidden);
  const float inverse_std = rsqrtf(fmaxf(0.0f, square_mean - mean * mean) + epsilon);
#pragma unroll
  for (int item = 0; item < kValuesPerLane; ++item) {
    const int column = lane + item * kWavefrontSize;
    output[base + column] = __float2half_rn((values[item] - mean) * inverse_std);
  }
}

template <int Hidden>
__global__ __launch_bounds__(kThreadsPerBlock) void residual_add_layer_norm_kernel(
    __half* __restrict__ residual_destination,
    const __half* __restrict__ residual,
    __half* __restrict__ output,
    int64_t rows,
    float epsilon) {
  constexpr int kValuesPerLane = Hidden / kWavefrontSize;
  const int lane = static_cast<int>(threadIdx.x) & (kWavefrontSize - 1);
  const int wave = static_cast<int>(threadIdx.x) / kWavefrontSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kRowsPerSoftmaxBlock + wave;
  if (row >= rows) {
    return;
  }

  const int64_t base = row * Hidden;
  float values[kValuesPerLane];
  float local_sum = 0.0f;
  float local_square_sum = 0.0f;
#pragma unroll
  for (int item = 0; item < kValuesPerLane; ++item) {
    const int column = lane + item * kWavefrontSize;
    const int64_t offset = base + column;
    const __half added = __hadd(
        residual_destination[offset],
        residual[offset]);
    residual_destination[offset] = added;
    const float value = __half2float(added);
    values[item] = value;
    local_sum += value;
    local_square_sum = fmaf(value, value, local_square_sum);
  }

  const float mean = wave_reduce_sum(local_sum) * (1.0f / Hidden);
  const float square_mean =
      wave_reduce_sum(local_square_sum) * (1.0f / Hidden);
  const float inverse_std = rsqrtf(fmaxf(0.0f, square_mean - mean * mean) + epsilon);
#pragma unroll
  for (int item = 0; item < kValuesPerLane; ++item) {
    const int column = lane + item * kWavefrontSize;
    output[base + column] = __float2half_rn((values[item] - mean) * inverse_std);
  }
}

template <int Hidden>
__global__ __launch_bounds__(kThreadsPerBlock) void index_select_tokens_kernel(
    const __half* __restrict__ input,
    const int32_t* __restrict__ index,
    __half* __restrict__ output,
    int64_t input_tokens,
    int64_t output_tokens,
    int64_t total_output_tokens) {
  constexpr int kPairs = Hidden / 2;
  const int lane = static_cast<int>(threadIdx.x) & (kWavefrontSize - 1);
  const int wave = static_cast<int>(threadIdx.x) / kWavefrontSize;
  const int64_t output_token =
      static_cast<int64_t>(blockIdx.x) * kRowsPerSoftmaxBlock + wave;
  if (output_token >= total_output_tokens) {
    return;
  }

  const int64_t batch = output_token / output_tokens;
  const int64_t output_in_batch = output_token - batch * output_tokens;
  const int32_t source_in_batch = index[output_in_batch];
  auto* output_pairs = reinterpret_cast<__half2*>(output);
  const auto* input_pairs = reinterpret_cast<const __half2*>(input);
  const int64_t output_pair_base = output_token * kPairs;
  if (source_in_batch < 0) {
    const __half2 zero = __float2half2_rn(0.0f);
#pragma unroll
    for (int pair = lane; pair < kPairs; pair += kWavefrontSize) {
      output_pairs[output_pair_base + pair] = zero;
    }
    return;
  }
  const int64_t input_pair_base =
      (batch * input_tokens + source_in_batch) * kPairs;
#pragma unroll
  for (int pair = lane; pair < kPairs; pair += kWavefrontSize) {
    output_pairs[output_pair_base + pair] = input_pairs[input_pair_base + pair];
  }
}

template <int Hidden>
__global__ __launch_bounds__(kThreadsPerBlock) void layer_norm_gather_kernel(
    const __half* __restrict__ input,
    const int32_t* __restrict__ index,
    __half* __restrict__ output,
    int64_t input_tokens,
    int64_t output_tokens,
    int64_t total_output_tokens,
    float epsilon) {
  constexpr int kValuesPerLane = Hidden / kWavefrontSize;
  const int lane = static_cast<int>(threadIdx.x) & (kWavefrontSize - 1);
  const int wave = static_cast<int>(threadIdx.x) / kWavefrontSize;
  const int64_t output_token =
      static_cast<int64_t>(blockIdx.x) * kRowsPerSoftmaxBlock + wave;
  if (output_token >= total_output_tokens) {
    return;
  }
  const int64_t batch = output_token / output_tokens;
  const int64_t output_in_batch = output_token - batch * output_tokens;
  const int32_t source_in_batch = index[output_in_batch];
  const int64_t output_base = output_token * Hidden;
  if (source_in_batch < 0) {
#pragma unroll
    for (int item = 0; item < kValuesPerLane; ++item) {
      output[output_base + lane + item * kWavefrontSize] = __float2half_rn(0.0f);
    }
    return;
  }

  const int64_t input_base =
      (batch * input_tokens + source_in_batch) * Hidden;
  float values[kValuesPerLane];
  float local_sum = 0.0f;
  float local_square_sum = 0.0f;
#pragma unroll
  for (int item = 0; item < kValuesPerLane; ++item) {
    const int column = lane + item * kWavefrontSize;
    const float value = __half2float(input[input_base + column]);
    values[item] = value;
    local_sum += value;
    local_square_sum = fmaf(value, value, local_square_sum);
  }
  const float mean = wave_reduce_sum(local_sum) * (1.0f / Hidden);
  const float square_mean =
      wave_reduce_sum(local_square_sum) * (1.0f / Hidden);
  const float inverse_std = rsqrtf(fmaxf(0.0f, square_mean - mean * mean) + epsilon);
#pragma unroll
  for (int item = 0; item < kValuesPerLane; ++item) {
    const int column = lane + item * kWavefrontSize;
    output[output_base + column] =
        __float2half_rn((values[item] - mean) * inverse_std);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void affine_fp16_to_fp32_kernel(
    const __half* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    float2* __restrict__ output,
    int64_t rows,
    int width_pairs,
    int channels,
    int height,
    int64_t input_stride_batch,
    int64_t input_stride_channel,
    int64_t input_stride_height) {
  const int64_t pair_index =
      static_cast<int64_t>(blockIdx.x) * kThreadsPerBlock + threadIdx.x;
  const int64_t total_pairs = rows * width_pairs;
  if (pair_index >= total_pairs) {
    return;
  }

  const int64_t row = pair_index / width_pairs;
  const int pair_in_row = static_cast<int>(pair_index - row * width_pairs);
  const int64_t batch_channel = row / height;
  const int64_t height_index = row - batch_channel * height;
  const int channel = static_cast<int>(batch_channel % channels);
  const int64_t batch = batch_channel / channels;
  const int64_t input_offset =
      batch * input_stride_batch + channel * input_stride_channel +
      height_index * input_stride_height + pair_in_row * 2;

  const __half2 pair = *reinterpret_cast<const __half2*>(input + input_offset);
  const float2 values = __half22float2(pair);
  const float scale = std[channel];
  const float offset = mean[channel];
  float2 result;
  result.x = __fadd_rn(__fmul_rn(values.x, scale), offset);
  result.y = __fadd_rn(__fmul_rn(values.y, scale), offset);
  output[pair_index] = result;
}

}  // namespace

at::Tensor launch_fused_bias_mask_add_hip(
    at::Tensor attn,
    const at::Tensor& bias,
    const at::Tensor& mask) {
  const int64_t batch = attn.size(0);
  const int64_t heads64 = attn.size(1);
  const int64_t pressure64 = attn.size(2);
  const int64_t planes = batch * heads64 * pressure64;
  TORCH_CHECK(
      planes > 0 && planes <= std::numeric_limits<uint32_t>::max(),
      "attention plane count exceeds the one-dimensional HIP grid");

  c10::hip::HIPGuard device_guard(attn.get_device());
  const hipStream_t stream =
      c10::hip::getCurrentHIPStream(attn.get_device()).stream();
  hipLaunchKernelGGL(
      fused_bias_mask_add_kernel,
      dim3(static_cast<uint32_t>(planes)),
      dim3(kThreadsPerBlock),
      0,
      stream,
      reinterpret_cast<__half*>(attn.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
      mask.data_ptr<int8_t>(),
      static_cast<int>(heads64),
      static_cast<int>(pressure64),
      static_cast<int>(mask.size(0)));
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return attn;
}

at::Tensor launch_layer_norm_no_affine_hip(
    const at::Tensor& input,
    double epsilon) {
  const int64_t hidden = input.size(-1);
  const int64_t rows = input.numel() / hidden;
  const int64_t blocks =
      (rows + kRowsPerSoftmaxBlock - 1) / kRowsPerSoftmaxBlock;
  TORCH_CHECK(
      blocks > 0 && blocks <= std::numeric_limits<uint32_t>::max(),
      "layer norm row count exceeds the one-dimensional HIP grid");

  at::Tensor output = at::empty_like(input);
  c10::hip::HIPGuard device_guard(input.get_device());
  const hipStream_t stream =
      c10::hip::getCurrentHIPStream(input.get_device()).stream();
  if (hidden == 192) {
    hipLaunchKernelGGL(
        (layer_norm_no_affine_kernel<192>),
        dim3(static_cast<uint32_t>(blocks)),
        dim3(kThreadsPerBlock),
        0,
        stream,
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        rows,
        static_cast<float>(epsilon));
  } else if (hidden == 384) {
    hipLaunchKernelGGL(
        (layer_norm_no_affine_kernel<384>),
        dim3(static_cast<uint32_t>(blocks)),
        dim3(kThreadsPerBlock),
        0,
        stream,
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        rows,
        static_cast<float>(epsilon));
  } else {
    hipLaunchKernelGGL(
        (layer_norm_no_affine_kernel<768>),
        dim3(static_cast<uint32_t>(blocks)),
        dim3(kThreadsPerBlock),
        0,
        stream,
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        rows,
        static_cast<float>(epsilon));
  }
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor launch_residual_add_layer_norm_hip(
    at::Tensor residual_destination,
    const at::Tensor& residual,
    double epsilon) {
  const int64_t hidden = residual_destination.size(-1);
  const int64_t rows = residual_destination.numel() / hidden;
  const int64_t blocks =
      (rows + kRowsPerSoftmaxBlock - 1) / kRowsPerSoftmaxBlock;
  TORCH_CHECK(
      blocks > 0 && blocks <= std::numeric_limits<uint32_t>::max(),
      "residual LayerNorm row count exceeds the one-dimensional HIP grid");

  at::Tensor output = at::empty_like(residual_destination);
  c10::hip::HIPGuard device_guard(residual_destination.get_device());
  const hipStream_t stream = c10::hip::getCurrentHIPStream(
      residual_destination.get_device()).stream();
  if (hidden == 192) {
    hipLaunchKernelGGL(
        (residual_add_layer_norm_kernel<192>),
        dim3(static_cast<uint32_t>(blocks)),
        dim3(kThreadsPerBlock),
        0,
        stream,
        reinterpret_cast<__half*>(residual_destination.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        rows,
        static_cast<float>(epsilon));
  } else {
    hipLaunchKernelGGL(
        (residual_add_layer_norm_kernel<384>),
        dim3(static_cast<uint32_t>(blocks)),
        dim3(kThreadsPerBlock),
        0,
        stream,
        reinterpret_cast<__half*>(residual_destination.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        rows,
        static_cast<float>(epsilon));
  }
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor launch_index_select_tokens_hip(
    const at::Tensor& input,
    const at::Tensor& index) {
  const int64_t batch = input.size(0);
  const int64_t input_tokens = input.size(1);
  const int64_t hidden = input.size(2);
  const int64_t output_tokens = index.numel();
  const int64_t total_output_tokens = batch * output_tokens;
  const int64_t blocks =
      (total_output_tokens + kRowsPerSoftmaxBlock - 1) /
      kRowsPerSoftmaxBlock;
  TORCH_CHECK(
      blocks > 0 && blocks <= std::numeric_limits<uint32_t>::max(),
      "token gather count exceeds the one-dimensional HIP grid");

  at::Tensor output = at::empty(
      {batch, output_tokens, hidden},
      input.options());
  c10::hip::HIPGuard device_guard(input.get_device());
  const hipStream_t stream =
      c10::hip::getCurrentHIPStream(input.get_device()).stream();
  if (hidden == 192) {
    hipLaunchKernelGGL(
        (index_select_tokens_kernel<192>),
        dim3(static_cast<uint32_t>(blocks)),
        dim3(kThreadsPerBlock),
        0,
        stream,
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        index.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        input_tokens,
        output_tokens,
        total_output_tokens);
  } else {
    hipLaunchKernelGGL(
        (index_select_tokens_kernel<384>),
        dim3(static_cast<uint32_t>(blocks)),
        dim3(kThreadsPerBlock),
        0,
        stream,
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        index.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        input_tokens,
        output_tokens,
        total_output_tokens);
  }
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor launch_layer_norm_gather_hip(
    const at::Tensor& input,
    const at::Tensor& index,
    double epsilon) {
  const int64_t batch = input.size(0);
  const int64_t input_tokens = input.size(1);
  const int64_t hidden = input.size(2);
  const int64_t output_tokens = index.numel();
  const int64_t total_output_tokens = batch * output_tokens;
  const int64_t blocks =
      (total_output_tokens + kRowsPerSoftmaxBlock - 1) /
      kRowsPerSoftmaxBlock;
  TORCH_CHECK(
      blocks > 0 && blocks <= std::numeric_limits<uint32_t>::max(),
      "LayerNorm gather count exceeds the one-dimensional HIP grid");
  at::Tensor output = at::empty(
      {batch, output_tokens, hidden},
      input.options());
  c10::hip::HIPGuard device_guard(input.get_device());
  const hipStream_t stream =
      c10::hip::getCurrentHIPStream(input.get_device()).stream();
  if (hidden == 192) {
    hipLaunchKernelGGL(
        (layer_norm_gather_kernel<192>),
        dim3(static_cast<uint32_t>(blocks)),
        dim3(kThreadsPerBlock),
        0,
        stream,
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        index.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        input_tokens,
        output_tokens,
        total_output_tokens,
        static_cast<float>(epsilon));
  } else {
    hipLaunchKernelGGL(
        (layer_norm_gather_kernel<384>),
        dim3(static_cast<uint32_t>(blocks)),
        dim3(kThreadsPerBlock),
        0,
        stream,
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        index.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        input_tokens,
        output_tokens,
        total_output_tokens,
        static_cast<float>(epsilon));
  }
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor launch_affine_fp16_to_fp32_hip(
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& std,
    at::Tensor output) {
  const int64_t rows = input.size(0) * input.size(1) * input.size(2);
  const int64_t width_pairs64 = input.size(3) / 2;
  const int64_t total_pairs = rows * width_pairs64;
  const int64_t blocks = (total_pairs + kThreadsPerBlock - 1) / kThreadsPerBlock;
  TORCH_CHECK(
      blocks > 0 && blocks <= std::numeric_limits<uint32_t>::max(),
      "affine element count exceeds the one-dimensional HIP grid");

  c10::hip::HIPGuard device_guard(input.get_device());
  const hipStream_t stream =
      c10::hip::getCurrentHIPStream(input.get_device()).stream();
  hipLaunchKernelGGL(
      affine_fp16_to_fp32_kernel,
      dim3(static_cast<uint32_t>(blocks)),
      dim3(kThreadsPerBlock),
      0,
      stream,
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      mean.data_ptr<float>(),
      std.data_ptr<float>(),
      reinterpret_cast<float2*>(output.data_ptr<float>()),
      rows,
      static_cast<int>(width_pairs64),
      static_cast<int>(input.size(1)),
      static_cast<int>(input.size(2)),
      input.stride(0),
      input.stride(1),
      input.stride(2));
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return output;
}

