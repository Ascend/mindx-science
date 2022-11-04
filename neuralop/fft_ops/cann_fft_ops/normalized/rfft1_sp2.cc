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

#include "rfft1_sp2.h"

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
#define N2 2

using namespace std;
using namespace Eigen;

namespace {
const char *kRFFT1_SP2 = "RFFT1_SP2";
const uint32_t kOutputNum = 2;
const uint32_t kInputNum = 1;
}  // namespace

namespace aicpu {
uint32_t RFFT1SP2CpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "[%s] check input and output failed.", kRFFT1_SP2);
  vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((shape_x.size() >= 1), KERNEL_STATUS_PARAM_INVALID,
                     "Input must be at least rank 1, got [%zu].",
                     shape_x.size())
  DataType data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    case DT_FLOAT:
      return RFFT1SP2Compute<float, complex<float>>(ctx);
    case DT_DOUBLE:
      return RFFT1SP2Compute<double, complex<double>>(ctx);
    default:
      KERNEL_LOG_ERROR("RFFT1_SP2 kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename C>
uint32_t RFFT1SP2ComputeSingleBatch(CpuKernelContext &ctx, int64_t batch_idx) {
  const int64_t FFTDim = 1;
  const auto axes = ArrayXi::LinSpaced(FFTDim, 0, FFTDim - 1);
  vector<int64_t> s = ctx.GetAttr("s")->GetListInt();
  vector<int64_t> modes = ctx.GetAttr("modes")->GetListInt();
  vector<int64_t> x_shape = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  int64_t dim1 = x_shape.at(x_shape.size() - 1);
  int64_t num_x_batch = dim1;
  int64_t num_y_batch = modes[0];
  auto x_ptr = reinterpret_cast<T *>(ctx.Input(0)->GetData()) + batch_idx * num_x_batch;
  auto y_re_ptr = reinterpret_cast<T *>(ctx.Output(0)->GetData()) + batch_idx * num_y_batch;
  auto y_im_ptr = reinterpret_cast<T *>(ctx.Output(1)->GetData()) + batch_idx * num_y_batch;
  TensorMap<Eigen::Tensor<T, 1, RowMajor>> x_tensor(x_ptr, dim1);
  x_tensor = x_tensor / T(s[0]);
  // Compute the full FFT using a temporary tensor.
  Eigen::Tensor<C, 1, RowMajor> fft_temp(s[0]);
  fft_temp = x_tensor.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(axes);
  DSizes<DenseIndex, FFTDim> slice_start{1};
  DSizes<DenseIndex, FFTDim> slice_sizes{s[0] - s[0] / 2 - 1};
  fft_temp.slice(slice_start, slice_sizes) = T(N2) * fft_temp.slice(slice_start, slice_sizes);
  // Slice to shape (modes[0], )
  DSizes<DenseIndex, FFTDim> modes_start{0};
  DSizes<DenseIndex, FFTDim> modes_sizes{modes[0]};
  TensorMap<Eigen::Tensor<T, 1, RowMajor>> y_re_tensor(y_re_ptr, modes[0]);
  TensorMap<Eigen::Tensor<T, 1, RowMajor>> y_im_tensor(y_im_ptr, modes[0]);
  y_re_tensor = fft_temp.slice(modes_start, modes_sizes).real();
  y_im_tensor = fft_temp.slice(modes_start, modes_sizes).imag();
  return KERNEL_STATUS_OK;
}

template <typename T, typename C>
uint32_t RFFT1SP2CpuKernel::RFFT1SP2Compute(CpuKernelContext &ctx) {
  const int64_t FFTDim = 1;
  vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  int64_t dim0 = 1;
  for (size_t i = 0; i < shape_x.size() - FFTDim; i++) { dim0 *= shape_x.at(i); }
  uint32_t min_core = 1;
  uint32_t max_core = std::max(min_core, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
  max_core = min(max_core, (uint32_t)dim0);
  auto shard_rfft1_sp2 = [&](int64_t start, int64_t end) {
    for (int64_t batch_idx = start; batch_idx < end; batch_idx++) {
      RFFT1SP2ComputeSingleBatch<T, C>(ctx, batch_idx);
    }
  };
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, dim0, dim0 / max_core, shard_rfft1_sp2),
                      "RFFT1_SP2 Compute failed.")
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kRFFT1_SP2, RFFT1SP2CpuKernel);
}  // namespace aicpu
