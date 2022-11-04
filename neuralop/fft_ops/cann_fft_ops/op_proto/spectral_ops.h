/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

/*!
 * \file spectral_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SPECTRAL_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SPECTRAL_OPS_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {
// FFT1D op
REG_OP(FFT1)
    .INPUT(x_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(FFT1)

// IFFT1D op
REG_OP(IFFT1)
    .INPUT(x_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(IFFT1)

// FFT2D op
REG_OP(FFT2)
    .INPUT(x_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(FFT2)

// IFFT2D op
REG_OP(IFFT2)
    .INPUT(x_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(IFFT2)

// FFT3D op
REG_OP(FFT3)
    .INPUT(x_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(FFT3)

// IFFT3D op
REG_OP(IFFT3)
    .INPUT(x_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(IFFT3)

// RFFT1D op
REG_OP(RFFT1)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(s, ListInt)
    .OP_END_FACTORY_REG(RFFT1)

// IRFFT1D op
REG_OP(IRFFT1)
    .INPUT(x_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(s, ListInt)
    .OP_END_FACTORY_REG(IRFFT1)

// RFFT2D op
REG_OP(RFFT2)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(s, ListInt)
    .OP_END_FACTORY_REG(RFFT2)

// IRFFT2D op
REG_OP(IRFFT2)
    .INPUT(x_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(s, ListInt)
    .OP_END_FACTORY_REG(IRFFT2)

// RFFT3D op
REG_OP(RFFT3)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(s, ListInt)
    .OP_END_FACTORY_REG(RFFT3)

// IRFFT3D op
REG_OP(IRFFT3)
    .INPUT(x_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(s, ListInt)
    .OP_END_FACTORY_REG(IRFFT3)

// RFFT1D_SP op
REG_OP(RFFT1_SP)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(s, ListInt)
    .REQUIRED_ATTR(modes, ListInt)
    .OP_END_FACTORY_REG(RFFT1_SP)

// RFFT2D_SP op
REG_OP(RFFT2_SP)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re_1, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re_2, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im_1, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im_2, TensorType({DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(s, ListInt)
    .REQUIRED_ATTR(modes, ListInt)
    .OP_END_FACTORY_REG(RFFT2_SP)

// RFFT3D_SP op
REG_OP(RFFT3_SP)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re_1, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re_2, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re_3, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re_4, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im_1, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im_2, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im_3, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im_4, TensorType({DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(s, ListInt)
    .REQUIRED_ATTR(modes, ListInt)
    .OP_END_FACTORY_REG(RFFT3_SP)

// IFFT1D_SP op
REG_OP(IFFT1_SP)
    .INPUT(x_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(s, ListInt)
    .REQUIRED_ATTR(origin, ListInt)
    .OP_END_FACTORY_REG(IFFT1_SP)

// IFFT2D_SP op
REG_OP(IFFT2_SP)
    .INPUT(x_re_1, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_re_2, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im_1, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im_2, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(s, ListInt)
    .REQUIRED_ATTR(origin, ListInt)
    .OP_END_FACTORY_REG(IFFT2_SP)

// IFFT3D_SP op
REG_OP(IFFT3_SP)
    .INPUT(x_re_1, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_re_2, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_re_3, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_re_4, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im_1, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im_2, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im_3, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im_4, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(s, ListInt)
    .REQUIRED_ATTR(origin, ListInt)
    .OP_END_FACTORY_REG(IFFT3_SP)

// RFFT1D_SP2 op
REG_OP(RFFT1_SP2)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(s, ListInt)
    .REQUIRED_ATTR(origin, ListInt)
    .REQUIRED_ATTR(modes, ListInt)
    .OP_END_FACTORY_REG(RFFT1_SP2)

// RFFT2D_SP2 op
REG_OP(RFFT2_SP2)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re_1, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re_2, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im_1, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im_2, TensorType({DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(s, ListInt)
    .REQUIRED_ATTR(origin, ListInt)
    .REQUIRED_ATTR(modes, ListInt)
    .OP_END_FACTORY_REG(RFFT2_SP2)

// RFFT3D_SP2 op
REG_OP(RFFT3_SP2)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re_1, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re_2, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re_3, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re_4, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im_1, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im_2, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im_3, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_im_4, TensorType({DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(s, ListInt)
    .REQUIRED_ATTR(origin, ListInt)
    .REQUIRED_ATTR(modes, ListInt)
    .OP_END_FACTORY_REG(RFFT3_SP2)

// IRFFT1D_SP op
REG_OP(IRFFT1_SP)
    .INPUT(x_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(s, ListInt)
    .REQUIRED_ATTR(origin, ListInt)
    .OP_END_FACTORY_REG(IRFFT1_SP)

// IRFFT2D_SP op
REG_OP(IRFFT2_SP)
    .INPUT(x_re_1, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_re_2, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im_1, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im_2, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(s, ListInt)
    .REQUIRED_ATTR(origin, ListInt)
    .OP_END_FACTORY_REG(IRFFT2_SP)

// IRFFT3D_SP op
REG_OP(IRFFT3_SP)
    .INPUT(x_re_1, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_re_2, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_re_3, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_re_4, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im_1, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im_2, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im_3, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(x_im_4, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_re, TensorType({DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(s, ListInt)
    .REQUIRED_ATTR(origin, ListInt)
    .OP_END_FACTORY_REG(IRFFT3_SP)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_SPECTRAL_OPS_H_