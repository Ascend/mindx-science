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

import math

import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
from mindspore.nn import CellList, Conv1d, Dense
from mindspore.common.initializer import Uniform as U

from architecture.basic_block import SpectralConv2d, _get_act

ms_prec = ms.float32
fft_prec = ms.float32


class FNO2d(nn.Cell):
    """
    Fourier Neural Operator
    """

    def __init__(self, modes1, modes2, width=32, *args, **kwargs):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9
        scale = math.sqrt(1 / width)
        self.conv0 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2)
        self.w_0 = nn.Conv2d(
            self.width,
            self.width,
            1,
            has_bias=True,
            weight_init=ms.common.initializer.Uniform(scale),
            bias_init=ms.common.initializer.Uniform(scale))
        self.w_1 = nn.Conv2d(
            self.width,
            self.width,
            1,
            has_bias=True,
            weight_init=ms.common.initializer.Uniform(scale),
            bias_init=ms.common.initializer.Uniform(scale))
        self.w_2 = nn.Conv2d(
            self.width,
            self.width,
            1,
            has_bias=True,
            weight_init=ms.common.initializer.Uniform(scale),
            bias_init=ms.common.initializer.Uniform(scale))
        self.w_3 = nn.Conv2d(
            self.width,
            self.width,
            1,
            has_bias=True,
            weight_init=ms.common.initializer.Uniform(scale),
            bias_init=ms.common.initializer.Uniform(scale))

        self.fc0 = nn.Dense(
            3,
            self.width,
            weight_init=ms.common.initializer.Uniform(
                1 / 3),
            bias_init=ms.common.initializer.Uniform(
                1 / 3))
        self.fc1 = nn.Dense(self.width, 128, weight_init=ms.common.initializer.Uniform(
            scale), bias_init=ms.common.initializer.Uniform(scale))
        self.fc2 = nn.Dense(
            128,
            1,
            weight_init=ms.common.initializer.Uniform(
                math.sqrt(
                    1 / 128)),
            bias_init=ms.common.initializer.Uniform(
                math.sqrt(
                    1 / 128)))

    def get_grid(self, shape):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = Tensor(np.linspace(0, 1, size_x), dtype=ms_prec)
        gridx = gridx.reshape(1, size_x, 1, 1)
        gridx = ms.numpy.tile(gridx, (batchsize, 1, size_y, 1))

        gridy = Tensor(np.linspace(0, 1, size_y), dtype=ms_prec)
        gridy = gridy.reshape(1, 1, size_y, 1)
        gridy = ms.numpy.tile(gridy, (batchsize, size_x, 1, 1))
        return ms.ops.Concat(-1)((gridx, gridy))

    def construct(self, x):

        grid = self.get_grid(x.shape)
        x = ms.ops.Concat(-1)((x, grid.astype(x.dtype)))

        x = self.fc0(x)
        x = ms.ops.Transpose()(x, (0, 3, 1, 2))
        pad_op = ms.nn.Pad(
            paddings=(
                (0, 0), (0, 0), (0, self.padding), (0, self.padding)))
        x = pad_op(x)

        x_1 = self.conv0(x)
        x_2 = self.w_0(x)
        x = x_1 + x_2
        x = ms.nn.GELU(approximate=False)(x)

        x_1 = self.conv1(x)
        x_2 = self.w_1(x)
        x = x_1 + x_2
        x = ms.nn.GELU(approximate=False)(x)

        x_1 = self.conv2(x)
        x_2 = self.w_2(x)
        x = x_1 + x_2
        x = ms.nn.GELU(approximate=False)(x)

        x_1 = self.conv2(x)
        x_2 = self.w_3(x)
        x = x_1 + x_2

        x = x[..., :-self.padding, :-self.padding]
        x = ms.ops.Transpose()(x, (0, 2, 3, 1))
        x = self.fc1(x)
        x = ms.nn.GELU(approximate=False)(x)
        x = self.fc2(x)
        return x


class FNO2dtime(nn.Cell):
    """
    Fourier Neural Operator
    """

    def __init__(self, modes1, modes2, width, t_in, *args, **kwargs):
        super(FNO2dtime, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        scale = math.sqrt(1 / width)
        in_dim = t_in + 2
        self.conv0 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2)
        self.w_0 = nn.Conv2d(
            self.width,
            self.width,
            1,
            has_bias=True,
            weight_init=ms.common.initializer.Uniform(scale),
            bias_init=ms.common.initializer.Uniform(scale))
        self.w_1 = nn.Conv2d(
            self.width,
            self.width,
            1,
            has_bias=True,
            weight_init=ms.common.initializer.Uniform(scale),
            bias_init=ms.common.initializer.Uniform(scale))
        self.w_2 = nn.Conv2d(
            self.width,
            self.width,
            1,
            has_bias=True,
            weight_init=ms.common.initializer.Uniform(scale),
            bias_init=ms.common.initializer.Uniform(scale))
        self.w_3 = nn.Conv2d(
            self.width,
            self.width,
            1,
            has_bias=True,
            weight_init=ms.common.initializer.Uniform(scale),
            bias_init=ms.common.initializer.Uniform(scale))

        self.fc0 = nn.Dense(
            in_dim, self.width, weight_init=ms.common.initializer.Uniform(
                math.sqrt(in_dim)), bias_init=ms.common.initializer.Uniform(
                math.sqrt(in_dim)))
        self.fc1 = nn.Dense(self.width, 128, weight_init=ms.common.initializer.Uniform(
            scale), bias_init=ms.common.initializer.Uniform(scale))
        self.fc2 = nn.Dense(
            128,
            1,
            weight_init=ms.common.initializer.Uniform(
                math.sqrt(
                    1 / 128)),
            bias_init=ms.common.initializer.Uniform(
                math.sqrt(
                    1 / 128)))

    def get_grid(self, shape):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = Tensor(np.linspace(0, 1, size_x), dtype=ms_prec)
        gridx = gridx.reshape(1, size_x, 1, 1)
        gridx = ms.numpy.tile(gridx, (batchsize, 1, size_y, 1))

        gridy = Tensor(np.linspace(0, 1, size_y), dtype=ms_prec)
        gridy = gridy.reshape(1, 1, size_y, 1)
        gridy = ms.numpy.tile(gridy, (batchsize, size_x, 1, 1))
        return ms.ops.Concat(-1)((gridx, gridy))

    def construct(self, x):
        grid = self.get_grid(x.shape)
        x = ms.ops.Concat(-1)((x, grid.astype(x.dtype)))
        x = self.fc0(x)
        x = ms.ops.Transpose()(x, (0, 3, 1, 2))

        x_1 = self.conv0(x)
        x_2 = self.w_0(x)
        x = x_1 + x_2
        x = ms.nn.GELU(approximate=False)(x)

        x_1 = self.conv1(x)
        x_2 = self.w_1(x)
        x = x_1 + x_2
        x = ms.nn.GELU(approximate=False)(x)

        x_1 = self.conv2(x)
        x_2 = self.w_2(x)
        x = x_1 + x_2
        x = ms.nn.GELU(approximate=False)(x)

        x_1 = self.conv2(x)
        x_2 = self.w_3(x)
        x = x_1 + x_2

        x = ms.ops.Transpose()(x, (0, 2, 3, 1))
        x = self.fc1(x)
        x = ms.nn.GELU(approximate=False)(x)
        x = self.fc2(x)
        return x


class FNN2d(nn.Cell):
    """
    Physics Informed Neural Operator
    """

    def __init__(self, modes1, modes2,
                 width=64, fc_dim=128,
                 layers=None,
                 in_dim=3, out_dim=1,
                 activation='tanh', *args, **kwargs):
        super(FNN2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.reshape = ms.ops.Reshape()
        self.transpose = ms.ops.Transpose()
        # input channel is 3: (a(x, y), x, y)
        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = Dense(
            in_dim,
            layers[0],
            weight_init=U(
                math.sqrt(
                    1 / in_dim)),
            bias_init=U(
                math.sqrt(
                    1 / in_dim)))

        self.sp_convs = CellList([SpectralConv2d(
            in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])

        all_s = []
        for in_size in layers:
            all_s.append(math.sqrt(1 / in_size))
        self.convs = CellList([Conv1d(in_, out_, 1, has_bias=True, weight_init=U(s), bias_init=U(
            s)) for in_, out_, s in zip(self.layers, self.layers[1:], all_s[1:])])

        self.fc1 = Dense(layers[-1],
                         fc_dim,
                         weight_init=U(math.sqrt(1 / layers[-1])),
                         bias_init=U(math.sqrt(1 / layers[-1])))
        self.fc2 = Dense(
            fc_dim,
            out_dim,
            weight_init=U(
                math.sqrt(
                    1 / fc_dim)),
            bias_init=U(
                math.sqrt(
                    1 / fc_dim)))
        self.activation = _get_act(activation)

    def construct(self, x):
        length = len(self.convs)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = self.transpose(x, (0, 3, 1, 2))

        for i, (speconv, conv) in enumerate(zip(self.sp_convs, self.convs)):
            x_1 = speconv(x)
            x_2 = conv(self.reshape(x, (batchsize, self.layers[i], -1)))
            x_2 = self.reshape(
                x_2, (batchsize, self.layers[i + 1], size_x, size_y))
            x = x_1 + x_2
            if i != length - 1:
                x = self.activation(x)
        x = self.transpose(x, (0, 2, 3, 1))
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
