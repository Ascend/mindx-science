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

#include "ifft2.h"

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
const char *kIFFT2 = "IFFT2";
const uint32_t kOutputNum = 2;
const uint32_t kInputNum = 2;
}  // namespace

namespace aicpu {
uint32_t IFFT2CpuKernel::Compute(CpuKernelContext &ctx) {
  const int32_t FFTDim = 2;
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "[%s] check input and output failed.", kIFFT2);
  vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((shape_x.size() >= FFTDim), KERNEL_STATUS_PARAM_INVALID,
                     "Input must be at least rank 2, got [%zu].",
                     shape_x.size())
  DataType data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    case DT_FLOAT:
      return IFFT2Compute<float>(ctx);
    case DT_DOUBLE:
      return IFFT2Compute<double>(ctx);
    default:
      KERNEL_LOG_ERROR("IFFT2 kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t IFFT2ComputeSingleBatch(CpuKernelContext &ctx, int64_t batch_idx) {
  const int32_t FFTDim = 2;
  const auto axes = ArrayXi::LinSpaced(FFTDim, 0, FFTDim - 1);
  vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  int64_t dim1 = shape_x.at(shape_x.size() - 2);
  int64_t dim2 = shape_x.at(shape_x.size() - 1);
  int64_t num_batch = dim1 * dim2;
  auto cur_x_re_ptr = reinterpret_cast<T *>(ctx.Input(0)->GetData()) + batch_idx * num_batch;
  auto cur_x_im_ptr = reinterpret_cast<T *>(ctx.Input(1)->GetData()) + batch_idx * num_batch;
  auto cur_y_re_ptr = reinterpret_cast<T *>(ctx.Output(0)->GetData()) + batch_idx * num_batch;
  auto cur_y_im_ptr = reinterpret_cast<T *>(ctx.Output(1)->GetData()) + batch_idx * num_batch;
  TensorMap<Eigen::Tensor<T, FFTDim, RowMajor>> cur_x_re_tensor(cur_x_re_ptr, dim1, dim2);
  TensorMap<Eigen::Tensor<T, FFTDim, RowMajor>> cur_x_im_tensor(cur_x_im_ptr, dim1, dim2);
  Eigen::Tensor<complex<T>, FFTDim, RowMajor> cur_x_tensor(dim1, dim2);
  for (int64_t i = 0; i < num_batch; i++) {
    cur_x_tensor.data()[i].real(cur_x_re_tensor.data()[i]);
    cur_x_tensor.data()[i].imag(cur_x_im_tensor.data()[i]);
  }
  Eigen::Tensor<complex<T>, FFTDim, RowMajor> cur_y_tensor(dim1, dim2);
  cur_y_tensor = cur_x_tensor.template fft<BothParts, FFT_REVERSE>(axes);
  TensorMap<Eigen::Tensor<T, FFTDim, RowMajor>> cur_y_re_tensor(cur_y_re_ptr, dim1, dim2);
  TensorMap<Eigen::Tensor<T, FFTDim, RowMajor>> cur_y_im_tensor(cur_y_im_ptr, dim1, dim2);
  cur_y_re_tensor = cur_y_tensor.real();
  cur_y_im_tensor = cur_y_tensor.imag();
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t IFFT2CpuKernel::IFFT2Compute(CpuKernelContext &ctx) {
  const int64_t FFTDim = 2;
  vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  int64_t dim0 = 1;
  for (size_t i = 0; i < shape_x.size() - FFTDim; i++) { dim0 *= shape_x.at(i); }
  uint32_t min_core = 1;
  uint32_t max_core = std::max(min_core, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
  max_core = min(max_core, (uint32_t)dim0);
  auto shard_ifft2 = [&](int64_t start, int64_t end) {
    for (int64_t batch_idx = start; batch_idx < end; batch_idx++) {
      IFFT2ComputeSingleBatch<T>(ctx, batch_idx);
    }
  };
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, dim0, dim0 / max_core, shard_ifft2),
                      "IFFT2 Compute failed.")
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kIFFT2, IFFT2CpuKernel);
}  // namespace aicpu
