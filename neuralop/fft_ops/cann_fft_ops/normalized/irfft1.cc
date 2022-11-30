/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "irfft1.h"

#include <iostream>
#include <algorithm>
#include <complex>
#include <map>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/FFT>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

using namespace std;
using namespace Eigen;

namespace {
const char *kIRFFT1 = "IRFFT1";
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
}  // namespace

namespace aicpu {
uint32_t IRFFT1CpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "[%s] check input and output failed.", kIRFFT1);
  vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((shape_x.size() >= 1), KERNEL_STATUS_PARAM_INVALID,
                     "Input must be at least rank 1, got [%zu].",
                     shape_x.size())
  DataType data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    case DT_FLOAT:
      return IRFFT1Compute<float, complex<float>>(ctx);
    case DT_DOUBLE:
      return IRFFT1Compute<double, complex<double>>(ctx);
    default:
      KERNEL_LOG_ERROR("IRFFT1 kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename C>
uint32_t IRFFT1ComputeSingleBatch(CpuKernelContext &ctx, int64_t batch_idx) {
  const int64_t FFTDim = 1;
  vector<int64_t> s = ctx.GetAttr("s")->GetListInt();
  vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  vector<int64_t> shape_y = ctx.Output(0)->GetTensorShape()->GetDimSizes();
  int64_t dim1_x = shape_x.at(shape_x.size() - 1);
  int64_t dim1_y = shape_y.at(shape_y.size() - 1);
  int64_t num_x_batch = dim1_x;
  int64_t num_y_batch = dim1_y;
  auto cur_x_re_ptr = reinterpret_cast<T *>(ctx.Input(0)->GetData()) + batch_idx * num_x_batch;
  auto cur_x_im_ptr = reinterpret_cast<T *>(ctx.Input(1)->GetData()) + batch_idx * num_x_batch;
  auto cur_y_ptr = reinterpret_cast<T *>(ctx.Output(0)->GetData()) + batch_idx * num_y_batch;
  TensorMap<Eigen::Tensor<T, 1, RowMajor>> cur_x_re_tensor(cur_x_re_ptr, dim1_x);
  TensorMap<Eigen::Tensor<T, 1, RowMajor>> cur_x_im_tensor(cur_x_im_ptr, dim1_x);
  Eigen::Tensor<C, 1, RowMajor> cur_x_tensor(dim1_x);
  for (int64_t i = 0; i < num_x_batch; i++) {
    cur_x_tensor.data()[i].real(cur_x_re_tensor.data()[i]);
    cur_x_tensor.data()[i].imag(cur_x_im_tensor.data()[i]);
  }
  TensorMap<Eigen::Tensor<T, 1, RowMajor>> cur_y_tensor(cur_y_ptr, dim1_y);
  DSizes<DenseIndex, FFTDim> in_sizes{s[0] / 2 + 1};
  Eigen::Tensor<C, 1, RowMajor> cur_full_fft(s[0]);
  auto temp_sizes = in_sizes;
  temp_sizes[FFTDim - 1] = s[FFTDim - 1] - in_sizes[FFTDim - 1];
  DSizes<DenseIndex, FFTDim> temp_indices2;
  temp_indices2[FFTDim - 1] = in_sizes[FFTDim - 1];
  const DSizes<DenseIndex, FFTDim> beigin_indices;
  DSizes<DenseIndex, FFTDim> temp_indices1;
  temp_indices1[FFTDim - 1] = 1;
  cur_full_fft.slice(beigin_indices, in_sizes) =
      cur_x_tensor.slice(beigin_indices, in_sizes);
  if (FFTDim > 1) {
    const auto outer_axes = ArrayXi::LinSpaced(FFTDim - 1, 0, FFTDim - 2);
    cur_full_fft.slice(beigin_indices, in_sizes) =
        cur_full_fft.slice(beigin_indices, in_sizes)
            .template fft<BothParts, FFT_REVERSE>(outer_axes);
  }
  Eigen::array<bool, FFTDim> reverse;
  for (auto i = 0; i < FFTDim; i++) {
    reverse[i] = (i == (FFTDim - 1));
  }
  if (temp_sizes[FFTDim - 1] != 0) {
    cur_full_fft.slice(temp_indices2, temp_sizes) =
        cur_full_fft.slice(temp_indices1, temp_sizes).reverse(reverse).conjugate();
  }
  auto inner_axis = Eigen::array<int, 1>{FFTDim - 1};
  cur_y_tensor = cur_full_fft.template fft<RealPart, FFT_REVERSE>(inner_axis);
  return KERNEL_STATUS_OK;
}

template <typename T, typename C>
uint32_t IRFFT1CpuKernel::IRFFT1Compute(CpuKernelContext &ctx) {
  const int64_t FFTDim = 1;
  vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  int64_t dim0 = 1;
  for (size_t i = 0; i < shape_x.size() - FFTDim; i++) { dim0 *= shape_x.at(i); }
  uint32_t min_core = 1;
  uint32_t max_core = std::max(min_core, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
  max_core = min(max_core, (uint32_t)dim0);
  auto shard_irfft1 = [&](int64_t start, int64_t end) {
    for (int64_t batch_idx = start; batch_idx < end; batch_idx++) {
      IRFFT1ComputeSingleBatch<T, C>(ctx, batch_idx);
    }
  };
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, dim0, dim0 / max_core, shard_irfft1),
                      "IRFFT1 Compute failed.")
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kIRFFT1, IRFFT1CpuKernel);
}  // namespace aicpu
