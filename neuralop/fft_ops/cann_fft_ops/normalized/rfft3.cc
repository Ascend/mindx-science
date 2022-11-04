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

#include "rfft3.h"

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
#define N3 3

using namespace std;
using namespace Eigen;

namespace {
const char *kRFFT3 = "RFFT3";
const uint32_t kOutputNum = 2;
const uint32_t kInputNum = 1;
}  // namespace

namespace aicpu {
uint32_t RFFT3CpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "[%s] check input and output failed.", kRFFT3);
  vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((shape_x.size() >= N3), KERNEL_STATUS_PARAM_INVALID,
                     "Input must be at least rank 3, got [%zu].",
                     shape_x.size())
  DataType data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    case DT_FLOAT:
      return RFFT3Compute<float, complex<float>>(ctx);
    case DT_DOUBLE:
      return RFFT3Compute<double, complex<double>>(ctx);
    default:
      KERNEL_LOG_ERROR("RFFT3 kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename C>
uint32_t RFFT3ComputeSingleBatch(CpuKernelContext &ctx, int64_t batch_idx) {
  const int64_t FFTDim = 3;
  vector<int64_t> s = ctx.GetAttr("s")->GetListInt();
  vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  vector<int64_t> shape_y = ctx.Output(0)->GetTensorShape()->GetDimSizes();
  int64_t dim1_x = shape_x.at(shape_x.size() - 3);
  int64_t dim2_x = shape_x.at(shape_x.size() - 2);
  int64_t dim3_x = shape_x.at(shape_x.size() - 1);
  int64_t dim1_y = shape_y.at(shape_y.size() - 3);
  int64_t dim2_y = shape_y.at(shape_y.size() - 2);
  int64_t dim3_y = shape_y.at(shape_y.size() - 1);
  int64_t num_x_batch = dim1_x * dim2_x * dim3_x;
  int64_t num_y_batch = dim1_y * dim2_y * dim3_y;
  auto cur_x_ptr = reinterpret_cast<T *>(ctx.Input(0)->GetData()) + batch_idx * num_x_batch;
  auto cur_y_re_ptr = reinterpret_cast<T *>(ctx.Output(0)->GetData()) + batch_idx * num_y_batch;
  auto cur_y_im_ptr = reinterpret_cast<T *>(ctx.Output(1)->GetData()) + batch_idx * num_y_batch;
  const auto axes = ArrayXi::LinSpaced(FFTDim, 0, FFTDim - 1);
  TensorMap<Eigen::Tensor<T, N3, RowMajor>> cur_x_tensor(cur_x_ptr, dim1_x, dim2_x, dim3_x);
  Eigen::Tensor<C, N3, RowMajor> cur_y_tensor(dim1_y, dim2_y, dim3_y);
  DSizes<DenseIndex, FFTDim> in_sizes{s[0], s[1], s[N2]};
  DSizes<DenseIndex, FFTDim> output_slice_sizes{dim1_y, dim2_y, dim3_y};
  DSizes<DenseIndex, FFTDim> zero_start_indices;
  // Compute the full FFT using a temporary tensor.
  Eigen::Tensor<C, N3, RowMajor> cur_full_fft(s[0], s[1], s[N2]);
  cur_full_fft = cur_x_tensor.slice(zero_start_indices, in_sizes)
                             .template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(axes);
  // Slice away the negative frequency components.
  cur_y_tensor = cur_full_fft.slice(zero_start_indices, output_slice_sizes);
  TensorMap<Eigen::Tensor<T, N3, RowMajor>> cur_y_re_tensor(cur_y_re_ptr, dim1_y, dim2_y, dim3_y);
  TensorMap<Eigen::Tensor<T, N3, RowMajor>> cur_y_im_tensor(cur_y_im_ptr, dim1_y, dim2_y, dim3_y);
  cur_y_re_tensor = cur_y_tensor.real();
  cur_y_im_tensor = cur_y_tensor.imag();
  return KERNEL_STATUS_OK;
}

template <typename T, typename C>
uint32_t RFFT3CpuKernel::RFFT3Compute(CpuKernelContext &ctx) {
  const int64_t FFTDim = 3;
  vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  int64_t dim0 = 1;
  for (size_t i = 0; i < shape_x.size() - FFTDim; i++) { dim0 *= shape_x.at(i); }
  uint32_t min_core = 1;
  uint32_t max_core = std::max(min_core, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
  max_core = min(max_core, (uint32_t)dim0);
  auto shard_rfft3 = [&](int64_t start, int64_t end) {
    for (int64_t batch_idx = start; batch_idx < end; batch_idx++) {
      RFFT3ComputeSingleBatch<T, C>(ctx, batch_idx);
    }
  };
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, dim0, dim0 / max_core, shard_rfft3),
                      "RFFT3 Compute failed.")
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kRFFT3, RFFT3CpuKernel);
}  // namespace aicpu
