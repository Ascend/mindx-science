# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np

from mindspore.ops import functional as F
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import nn

from architecture.fft_ops_1 import RFFT1_SP, RFFT2_SP, RFFT3_SP
from architecture.fft_ops_1 import IRFFT1_SP, IRFFT2_SP, IRFFT3_SP

np_prec = np.float32  # 建议和ms_prec一致
ms_prec = ms.float32
fft_prec = ms.float32


def _get_act(activation):
    if activation == 'tanh':
        func = ms.nn.Tanh()
    elif activation == 'gelu':
        func = ms.nn.GELU(approximate=False)
    elif activation == 'relu':
        func = ms.nn.ReLU
    else:
        raise ValueError(f'{activation} is not supported')
    return func


class SpectralConv1d(nn.Cell):

    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = (1 / (in_channels * out_channels))
        tensor1 = scale * mnp.rand(
            (in_channels, out_channels, modes), dtype=ms_prec)
        tensor2 = scale * mnp.rand(
            (in_channels, out_channels, modes), dtype=ms_prec)
        self.weights1 = ms.Parameter(tensor1)
        self.weights2 = ms.Parameter(tensor2)

    def construct(self, x):
        s = x.shape[2]
        # rfft1
        rfft1 = RFFT1_SP(s=(s, ), modes=(self.modes, ))
        net_prec = x.dtype
        x_ft_re, x_ft_im = rfft1(F.cast(x, fft_prec))
        x_ft_re = F.cast(x_ft_re, net_prec)
        x_ft_im = F.cast(x_ft_im, net_prec)

        # linear transformation
        m1 = self.compl_mul1d(x_ft_re, self.weights1)
        m2 = self.compl_mul1d(x_ft_im, self.weights2)
        m3 = self.compl_mul1d(x_ft_re, self.weights2)
        m4 = self.compl_mul1d(x_ft_im, self.weights1)

        irfft1 = IRFFT1_SP(s=(s, ), origin=(s // 2 + 1, ))

        # complex multimlication
        out_ft_re_trunc = F.cast(m1 - m2, net_prec)
        out_ft_im_trunc = F.cast(m3 + m4, net_prec)

        # irfft1
        x = irfft1(out_ft_re_trunc, out_ft_im_trunc)
        return x

    def compl_mul1d(self, inputs, weights):
        perm_1 = (2, 0, 1)
        perm_2 = (1, 2, 0)
        batmatmul = ms.ops.BatchMatMul()
        transpose = ms.ops.Transpose()
        inputs = transpose(inputs, perm_1)
        weights = transpose(weights, perm_1)
        output = batmatmul(inputs, weights)
        output = transpose(output, perm_2)
        return output


class SpectralConv2d(nn.Cell):

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2
        scale = (1 / (in_channels * out_channels))
        tensor1 = scale * mnp.rand(
            (in_channels, out_channels, modes1, modes2), dtype=ms_prec)
        tensor2 = scale * mnp.rand(
            (in_channels, out_channels, modes1, modes2), dtype=ms_prec)
        tensor1_ = scale * mnp.rand(
            (in_channels, out_channels, modes1, modes2), dtype=ms_prec)
        tensor2_ = scale * mnp.rand(
            (in_channels, out_channels, modes1, modes2), dtype=ms_prec)
        self.weights1 = ms.Parameter(tensor1)
        self.weights1_ = ms.Parameter(tensor1_)
        self.weights2 = ms.Parameter(tensor2)
        self.weights2_ = ms.Parameter(tensor2_)

    def construct(self, x):
        size1 = x.shape[-2]
        size2 = x.shape[-1]
        net_prec = x.dtype
        s = (size1, size2)
        # rfft2
        rfft2 = RFFT2_SP(s=s, modes=(self.modes1, self.modes2))

        x_ft_re1, x_ft_re2, x_ft_im1, x_ft_im2 = rfft2(F.cast(x, fft_prec))
        x_ft_re1 = F.cast(x_ft_re1, net_prec)
        x_ft_re2 = F.cast(x_ft_re2, net_prec)
        x_ft_im1 = F.cast(x_ft_im1, net_prec)
        x_ft_im2 = F.cast(x_ft_im2, net_prec)

        # linear transformation
        m1 = self.compl_mul2d(x_ft_re1, self.weights1)
        m1_ = self.compl_mul2d(x_ft_im1, self.weights1_)
        m2 = self.compl_mul2d(x_ft_re1, self.weights1_)
        m2_ = self.compl_mul2d(x_ft_im1, self.weights1)
        m3 = self.compl_mul2d(x_ft_re2, self.weights2)
        m3_ = self.compl_mul2d(x_ft_im2, self.weights2_)
        m4 = self.compl_mul2d(x_ft_re2, self.weights2_)
        m4_ = self.compl_mul2d(x_ft_im2, self.weights2)

        origin = (size1, size2 // 2 + 1)
        irfft2 = IRFFT2_SP(s, origin)

        out_ft_re_trunc_fr = F.cast(m1 - m1_, fft_prec)
        out_ft_im_trunc_fr = F.cast(m2 + m2_, fft_prec)
        out_ft_re_trunc_be = F.cast(m3 - m3_, fft_prec)
        out_ft_im_trunc_be = F.cast(m4 + m4_, fft_prec)
        x = irfft2(out_ft_re_trunc_fr, out_ft_re_trunc_be, out_ft_im_trunc_fr,
                   out_ft_im_trunc_be)

        return x

    def compl_mul2d(self, inputs, weights):
        perm_1 = (2, 3, 0, 1)
        batmatmul = ms.ops.BatchMatMul()
        transpose = ms.ops.Transpose()
        inputs = transpose(inputs, perm_1)
        weights = transpose(weights, perm_1)
        output = batmatmul(inputs, weights)
        output = transpose(output, perm_1)
        return output


class SpectralConv3d(nn.Cell):

    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = (1 / (in_channels * out_channels))
        tensor1 = scale * mnp.rand(
            (in_channels, out_channels, modes1, modes2, modes3),
            dtype=ms.float32)
        tensor2 = scale * mnp.rand(
            (in_channels, out_channels, modes1, modes2, modes3),
            dtype=ms.float32)
        tensor1_ = scale * mnp.rand(
            (in_channels, out_channels, modes1, modes2, modes3),
            dtype=ms.float32)
        tensor2_ = scale * mnp.rand(
            (in_channels, out_channels, modes1, modes2, modes3),
            dtype=ms.float32)
        tensor3 = scale * mnp.rand(
            (in_channels, out_channels, modes1, modes2, modes3),
            dtype=ms.float32)
        tensor4 = scale * mnp.rand(
            (in_channels, out_channels, modes1, modes2, modes3),
            dtype=ms.float32)
        tensor3_ = scale * mnp.rand(
            (in_channels, out_channels, modes1, modes2, modes3),
            dtype=ms.float32)
        tensor4_ = scale * mnp.rand(
            (in_channels, out_channels, modes1, modes2, modes3),
            dtype=ms.float32)
        self.weights1 = ms.Parameter(tensor1)
        self.weights1_ = ms.Parameter(tensor1_)
        self.weights2 = ms.Parameter(tensor2)
        self.weights2_ = ms.Parameter(tensor2_)
        self.weights3 = ms.Parameter(tensor3)
        self.weights3_ = ms.Parameter(tensor3_)
        self.weights4 = ms.Parameter(tensor4)
        self.weights4_ = ms.Parameter(tensor4_)

    def construct(self, x):
        size1 = x.shape[-3]
        size2 = x.shape[-2]
        size3 = x.shape[-1]
        net_prec = x.dtype
        s = (size1, size2, size3)
        modes = (self.modes1, self.modes2, self.modes3)
        rfft3 = RFFT3_SP(s, modes)
        x_ft_re1, x_ft_re2, x_ft_re3, x_ft_re4, x_ft_im1, x_ft_im2, x_ft_im3, x_ft_im4, = rfft3(
            F.cast(x, fft_prec))
        x_ft_re1 = F.cast(x_ft_re1, net_prec)
        x_ft_re2 = F.cast(x_ft_re2, net_prec)
        x_ft_re3 = F.cast(x_ft_re3, net_prec)
        x_ft_re4 = F.cast(x_ft_re4, net_prec)
        x_ft_im1 = F.cast(x_ft_im1, net_prec)
        x_ft_im2 = F.cast(x_ft_im2, net_prec)
        x_ft_im3 = F.cast(x_ft_im3, net_prec)
        x_ft_im4 = F.cast(x_ft_im4, net_prec)

        m1 = self.compl_mul3d(x_ft_re1, self.weights1)
        m2 = self.compl_mul3d(x_ft_re2, self.weights2)
        m3 = self.compl_mul3d(x_ft_re3, self.weights3)
        m4 = self.compl_mul3d(x_ft_re4, self.weights4)
        m5 = self.compl_mul3d(x_ft_re1, self.weights1_)
        m6 = self.compl_mul3d(x_ft_re2, self.weights2_)
        m7 = self.compl_mul3d(x_ft_re3, self.weights3_)
        m8 = self.compl_mul3d(x_ft_re4, self.weights4_)
        m1_ = self.compl_mul3d(x_ft_im1, self.weights1)
        m2_ = self.compl_mul3d(x_ft_im2, self.weights2)
        m3_ = self.compl_mul3d(x_ft_im3, self.weights3)
        m4_ = self.compl_mul3d(x_ft_im4, self.weights4)
        m5_ = self.compl_mul3d(x_ft_im1, self.weights1_)
        m6_ = self.compl_mul3d(x_ft_im2, self.weights2_)
        m7_ = self.compl_mul3d(x_ft_im3, self.weights3_)
        m8_ = self.compl_mul3d(x_ft_im4, self.weights4_)

        origin = (size1, size2, size3 // 2 + 1)
        irfft3 = IRFFT3_SP(s, origin)
        x = irfft3(F.cast(m1 - m1_, fft_prec), F.cast(m2 - m2_, fft_prec),
                   F.cast(m3 - m3_, fft_prec), F.cast(m4 - m4_, fft_prec),
                   F.cast(m5 + m5_, fft_prec), F.cast(m6 + m6_, fft_prec),
                   F.cast(m7 + m7_, fft_prec), F.cast(m8 + m8_, fft_prec))
        return x

    def compl_mul3d(self, inputs, weights):
        perm_1 = (2, 3, 4, 0, 1)
        perm_2 = (3, 4, 0, 1, 2)
        batmatmul = ms.ops.BatchMatMul()
        transpose = ms.ops.Transpose()
        inputs = transpose(inputs, perm_1)
        weights = transpose(weights, perm_1)
        output = batmatmul(inputs, weights)
        output = transpose(output, perm_2)
        return output
