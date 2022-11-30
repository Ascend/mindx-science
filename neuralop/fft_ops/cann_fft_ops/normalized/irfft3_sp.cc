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

#include "irfft3_sp.h"

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
#define N3 3

using namespace std;
using namespace Eigen;

namespace {
const char *kIRFFT3_SP = "IRFFT3_SP";
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 8;
}  // namespace

namespace aicpu {
uint32_t IRFFT3SPCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "[%s] check input and output failed.", kIRFFT3_SP);
  vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((shape_x.size() >= N3), KERNEL_STATUS_PARAM_INVALID,
                     "Input must be at least rank 3, got [%zu].",
                     shape_x.size())
  DataType data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    case DT_FLOAT:
      return IRFFT3SPCompute<float, complex<float>>(ctx);
    case DT_DOUBLE:
      return IRFFT3SPCompute<double, complex<double>>(ctx);
    default:
      KERNEL_LOG_ERROR("IRFFT3_SP kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename C>
uint32_t IRFFT3SPComputeSingleBatch(CpuKernelContext &ctx, int64_t batch_idx) {
  const int64_t FFTDim = 3;
  vector<int64_t> s = ctx.GetAttr("s")->GetListInt();
  vector<int64_t> origin = ctx.GetAttr("origin")->GetListInt();
  vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  vector<int64_t> shape_y = ctx.Output(0)->GetTensorShape()->GetDimSizes();
  int64_t mode1 = shape_x.at(shape_x.size() - 3);
  int64_t mode2 = shape_x.at(shape_x.size() - 2);
  int64_t mode3 = shape_x.at(shape_x.size() - 1);
  int64_t dim1_y = shape_y.at(shape_y.size() - 3);
  int64_t dim2_y = shape_y.at(shape_y.size() - 2);
  int64_t dim3_y = shape_y.at(shape_y.size() - 1);
  int64_t num_x_batch = mode1 * mode2 * mode3;
  int64_t num_y_batch = dim1_y * dim2_y * dim3_y;
  auto x1_re_ptr = reinterpret_cast<T *>(ctx.Input(0)->GetData()) + batch_idx * num_x_batch;
  auto x2_re_ptr = reinterpret_cast<T *>(ctx.Input(1)->GetData()) + batch_idx * num_x_batch;
  auto x3_re_ptr = reinterpret_cast<T *>(ctx.Input(2)->GetData()) + batch_idx * num_x_batch;
  auto x4_re_ptr = reinterpret_cast<T *>(ctx.Input(3)->GetData()) + batch_idx * num_x_batch;
  auto x1_im_ptr = reinterpret_cast<T *>(ctx.Input(4)->GetData()) + batch_idx * num_x_batch;
  auto x2_im_ptr = reinterpret_cast<T *>(ctx.Input(5)->GetData()) + batch_idx * num_x_batch;
  auto x3_im_ptr = reinterpret_cast<T *>(ctx.Input(6)->GetData()) + batch_idx * num_x_batch;
  auto x4_im_ptr = reinterpret_cast<T *>(ctx.Input(7)->GetData()) + batch_idx * num_x_batch;
  auto y_ptr = reinterpret_cast<T *>(ctx.Output(0)->GetData()) + batch_idx * num_y_batch;
  TensorMap<Eigen::Tensor<T, N3, RowMajor>> x1_re_tensor(x1_re_ptr, mode1, mode2, mode3);
  TensorMap<Eigen::Tensor<T, N3, RowMajor>> x2_re_tensor(x2_re_ptr, mode1, mode2, mode3);
  TensorMap<Eigen::Tensor<T, N3, RowMajor>> x3_re_tensor(x3_re_ptr, mode1, mode2, mode3);
  TensorMap<Eigen::Tensor<T, N3, RowMajor>> x4_re_tensor(x4_re_ptr, mode1, mode2, mode3);
  TensorMap<Eigen::Tensor<T, N3, RowMajor>> x1_im_tensor(x1_im_ptr, mode1, mode2, mode3);
  TensorMap<Eigen::Tensor<T, N3, RowMajor>> x2_im_tensor(x2_im_ptr, mode1, mode2, mode3);
  TensorMap<Eigen::Tensor<T, N3, RowMajor>> x3_im_tensor(x3_im_ptr, mode1, mode2, mode3);
  TensorMap<Eigen::Tensor<T, N3, RowMajor>> x4_im_tensor(x4_im_ptr, mode1, mode2, mode3);
  Eigen::Tensor<C, N3, RowMajor> x1_tensor(mode1, mode2, mode3);
  Eigen::Tensor<C, N3, RowMajor> x2_tensor(mode1, mode2, mode3);
  Eigen::Tensor<C, N3, RowMajor> x3_tensor(mode1, mode2, mode3);
  Eigen::Tensor<C, N3, RowMajor> x4_tensor(mode1, mode2, mode3);
  for (int64_t i = 0; i < num_x_batch; i++) {
    x1_tensor.data()[i].real(x1_re_tensor.data()[i]);
    x1_tensor.data()[i].imag(x1_im_tensor.data()[i]);
    x2_tensor.data()[i].real(x2_re_tensor.data()[i]);
    x2_tensor.data()[i].imag(x2_im_tensor.data()[i]);
    x3_tensor.data()[i].real(x3_re_tensor.data()[i]);
    x3_tensor.data()[i].imag(x3_im_tensor.data()[i]);
    x4_tensor.data()[i].real(x4_re_tensor.data()[i]);
    x4_tensor.data()[i].imag(x4_im_tensor.data()[i]);
  }
  Eigen::Tensor<C, N3, RowMajor> x_tensor(origin[0], origin[1], origin[N2]);
  x_tensor.setConstant(0);
  DSizes<DenseIndex, FFTDim> modes_start1{ 0, 0, 0 };
  DSizes<DenseIndex, FFTDim> modes_start2{ origin[0] - mode1, 0, 0 };
  DSizes<DenseIndex, FFTDim> modes_start3{ 0, origin[1] - mode2, 0 };
  DSizes<DenseIndex, FFTDim> modes_start4{ origin[0] - mode1, origin[1] - mode2, 0 };
  DSizes<DenseIndex, FFTDim> modes_sizes{ mode1, mode2, mode3 };
  x_tensor.slice(modes_start1, modes_sizes) = x1_tensor;
  x_tensor.slice(modes_start2, modes_sizes) = x2_tensor;
  x_tensor.slice(modes_start3, modes_sizes) = x3_tensor;
  x_tensor.slice(modes_start4, modes_sizes) = x4_tensor;
  TensorMap<Eigen::Tensor<T, N3, RowMajor>> y_tensor(y_ptr, dim1_y, dim2_y, dim3_y);
  DSizes<DenseIndex, FFTDim> in_sizes{ s[0], s[1], s[N2] / N2 + 1 };
  Eigen::Tensor<C, N3, RowMajor> fft_temp(s[0], s[1], s[N2]);
  auto temp_sizes = in_sizes;
  temp_sizes[FFTDim - 1] = s[FFTDim - 1] - in_sizes[FFTDim - 1];
  DSizes<DenseIndex, FFTDim> temp_indices2;
  temp_indices2[FFTDim - 1] = in_sizes[FFTDim - 1];
  const DSizes<DenseIndex, FFTDim> beigin_indices;
  DSizes<DenseIndex, FFTDim> temp_indices1;
  temp_indices1[FFTDim - 1] = 1;
  fft_temp.slice(beigin_indices, in_sizes) =
      x_tensor.slice(beigin_indices, in_sizes);
  if (FFTDim > 1) {
    const auto outer_axes = ArrayXi::LinSpaced(FFTDim - 1, 0, FFTDim - 2);
    fft_temp.slice(beigin_indices, in_sizes) =
        fft_temp.slice(beigin_indices, in_sizes)
            .template fft<BothParts, FFT_REVERSE>(outer_axes);
  }
  Eigen::array<bool, FFTDim> reverse;
  for (auto i = 0; i < FFTDim; i++) {
    reverse[i] = (i == (FFTDim - 1));
  }
  if (temp_sizes[FFTDim - 1] != 0) {
    fft_temp.slice(temp_indices2, temp_sizes) =
        fft_temp.slice(temp_indices1, temp_sizes).reverse(reverse).conjugate();
  }
  auto inner_axis = Eigen::array<int, 1>{ FFTDim - 1 };
  y_tensor = fft_temp.template fft<RealPart, FFT_REVERSE>(inner_axis);
  return KERNEL_STATUS_OK;
}

template <typename T, typename C>
uint32_t IRFFT3SPCpuKernel::IRFFT3SPCompute(CpuKernelContext &ctx) {
  const int64_t FFTDim = 3;
  vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  int64_t dim0 = 1;
  for (size_t i = 0; i < shape_x.size() - FFTDim; i++) { dim0 *= shape_x.at(i); }
  uint32_t min_core = 1;
  uint32_t max_core = std::max(min_core, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
  max_core = min(max_core, (uint32_t)dim0);
  auto shard_irfft3_sp = [&](int64_t start, int64_t end) {
    for (int64_t batch_idx = start; batch_idx < end; batch_idx++) {
      IRFFT3SPComputeSingleBatch<T, C>(ctx, batch_idx);
    }
  };
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, dim0, dim0 / max_core, shard_irfft3_sp),
                      "IRFFT3_SP Compute failed.")
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kIRFFT3_SP, IRFFT3SPCpuKernel);
}  // namespace aicpu
