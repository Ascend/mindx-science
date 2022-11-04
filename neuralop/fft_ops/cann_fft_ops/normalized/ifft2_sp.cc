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

#include "ifft2_sp.h"

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
const char *kIFFT2_SP = "IFFT2_SP";
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 4;
}  // namespace

namespace aicpu {
const int32_t FFTDim = 2;
uint32_t IFFT2SPCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "[%s] check input and output failed.", kIFFT2_SP);
  vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((shape_x.size() >= FFTDim), KERNEL_STATUS_PARAM_INVALID,
                     "Input must be at least rank 2, got [%zu].",
                     shape_x.size())
  DataType data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    case DT_FLOAT:
      return IFFT2SPCompute<float>(ctx);
    case DT_DOUBLE:
      return IFFT2SPCompute<double>(ctx);
    default:
      KERNEL_LOG_ERROR("IFFT2_SP kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t IFFT2SPComputeSingleBatch(CpuKernelContext &ctx, int64_t batch_idx) {
  const int32_t FFTDim = 2;
  const auto axes = ArrayXi::LinSpaced(FFTDim, 0, FFTDim - 1);
  vector<int64_t> s = ctx.GetAttr("s")->GetListInt();
  vector<int64_t> origin = ctx.GetAttr("origin")->GetListInt();
  vector<int64_t> x_shape = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  int64_t mode1 = x_shape.at(x_shape.size() - 2);
  int64_t mode2 = x_shape.at(x_shape.size() - 1);
  int64_t num_batch_x = mode1 * mode2;
  int64_t num_batch_y = origin[0] * origin[1];
  auto x_re1_ptr = reinterpret_cast<T *>(ctx.Input(0)->GetData()) + batch_idx * num_batch_x;
  auto x_re2_ptr = reinterpret_cast<T *>(ctx.Input(1)->GetData()) + batch_idx * num_batch_x;
  auto x_im1_ptr = reinterpret_cast<T *>(ctx.Input(2)->GetData()) + batch_idx * num_batch_x;
  auto x_im2_ptr = reinterpret_cast<T *>(ctx.Input(3)->GetData()) + batch_idx * num_batch_x;
  auto y_re_ptr = reinterpret_cast<T *>(ctx.Output(0)->GetData()) + batch_idx * num_batch_y;
  TensorMap<Eigen::Tensor<T, FFTDim, RowMajor>> x_re1_tensor(x_re1_ptr, mode1, mode2);
  TensorMap<Eigen::Tensor<T, FFTDim, RowMajor>> x_re2_tensor(x_re2_ptr, mode1, mode2);
  TensorMap<Eigen::Tensor<T, FFTDim, RowMajor>> x_im1_tensor(x_im1_ptr, mode1, mode2);
  TensorMap<Eigen::Tensor<T, FFTDim, RowMajor>> x_im2_tensor(x_im2_ptr, mode1, mode2);
  Eigen::Tensor<complex<T>, FFTDim, RowMajor> x1_tensor(mode1, mode2);
  Eigen::Tensor<complex<T>, FFTDim, RowMajor> x2_tensor(mode1, mode2);
  for (int64_t i = 0; i < num_batch_x; i++) {
    x1_tensor.data()[i].real(s[0] * s[1] * x_re1_tensor.data()[i]);
    x1_tensor.data()[i].imag(s[0] * s[1] * x_im1_tensor.data()[i]);
    x2_tensor.data()[i].real(s[0] * s[1] * x_re2_tensor.data()[i]);
    x2_tensor.data()[i].imag(s[0] * s[1] * x_im2_tensor.data()[i]);
  }
  // recover shape (s[0], s[1]) and implement IFFT2D
  Eigen::Tensor<complex<T>, FFTDim, RowMajor> s_tensor(s[0], s[1]);
  s_tensor.setConstant(0);
  DSizes<DenseIndex, FFTDim> modes_start1{0, 0};
  DSizes<DenseIndex, FFTDim> modes_start2{s[0] - mode1, 0};
  DSizes<DenseIndex, FFTDim> modes_sizes{mode1, mode2};
  s_tensor.slice(modes_start1, modes_sizes) = x1_tensor;
  s_tensor.slice(modes_start2, modes_sizes) = x2_tensor;
  Eigen::Tensor<T, FFTDim, RowMajor> s_ifft_tensor(s[0], s[1]);
  s_ifft_tensor = s_tensor.template fft<RealPart, FFT_REVERSE>(axes);
  // recover shape (origin[0], origin[1])
  TensorMap<Eigen::Tensor<T, FFTDim, RowMajor>> y_re_tensor(y_re_ptr, origin[0], origin[1]);
  if ((origin[0] - s[0]) || (origin[1] - s[1])) {
    y_re_tensor.setConstant(0);
    DSizes<DenseIndex, FFTDim> s_start{0, 0};
    DSizes<DenseIndex, FFTDim> s_sizes{s[0], s[1]};
    y_re_tensor.slice(s_start, s_sizes) = s_ifft_tensor;
    return KERNEL_STATUS_OK;
  } else {
    y_re_tensor = s_ifft_tensor;
    return KERNEL_STATUS_OK;
  }
}

template <typename T>
uint32_t IFFT2SPCpuKernel::IFFT2SPCompute(CpuKernelContext &ctx) {
  const int64_t FFTDim = 2;
  vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  int64_t dim0 = 1;
  for (size_t i = 0; i < shape_x.size() - FFTDim; i++) { dim0 *= shape_x.at(i); }
  uint32_t min_core = 1;
  uint32_t max_core = std::max(min_core, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
  max_core = min(max_core, (uint32_t)dim0);
  auto shard_ifft2_sp = [&](int64_t start, int64_t end) {
    for (int64_t batch_idx = start; batch_idx < end; batch_idx++) {
      IFFT2SPComputeSingleBatch<T>(ctx, batch_idx);
    }
  };
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, dim0, dim0 / max_core, shard_ifft2_sp),
                      "IFFT2_SP Compute failed.")
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kIFFT2_SP, IFFT2SPCpuKernel);
}  // namespace aicpu
