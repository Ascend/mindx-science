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

import mindspore as ms
from mindspore import nn, Tensor
from mindspore.common.initializer import Uniform as U

from architecture.basic_block import SpectralConv1d


class FNO1d(nn.Cell):
    """
    Fourier Neural Operator
    """

    def __init__(self, modes, width=64, *args, **kwargs):
        super(FNO1d, self).__init__()
        self.modes = modes
        self.width = width
        scale = math.sqrt(1 / width)
        self.conv0 = SpectralConv1d(width, width, modes)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.conv3 = SpectralConv1d(width, width, modes)
        self.w_0 = nn.Conv1d(
            width,
            width,
            1,
            has_bias=True,
            weight_init=U(scale),
            bias_init=U(scale))
        self.w_1 = nn.Conv1d(
            width,
            width,
            1,
            has_bias=True,
            weight_init=U(scale),
            bias_init=U(scale))
        self.w_2 = nn.Conv1d(
            width,
            width,
            1,
            has_bias=True,
            weight_init=U(scale),
            bias_init=U(scale))
        self.w_3 = nn.Conv1d(
            width,
            width,
            1,
            has_bias=True,
            weight_init=U(scale),
            bias_init=U(scale))
        self.fc0 = nn.Dense(
            2,
            width,
            weight_init=U(
                math.sqrt(
                    1 / 2)),
            bias_init=U(
                math.sqrt(
                    1 / 2)))
        self.fc1 = nn.Dense(
            width,
            128,
            weight_init=U(scale),
            bias_init=U(scale))
        self.fc2 = nn.Dense(
            128,
            1,
            weight_init=U(
                math.sqrt(
                    1 / 128)),
            bias_init=U(
                math.sqrt(
                    1 / 128)))

    def get_grid(self, shape):
        batchsize, s = shape[0], shape[1]
        gridx = Tensor(ms.numpy.linspace(0, 1, s))
        gridx = gridx.reshape(1, s, 1)
        gridx = ms.numpy.tile(gridx, (batchsize, 1, 1))
        return gridx

    def construct(self, x):
        grid = self.get_grid(x.shape)
        x = ms.ops.Concat(-1)((x, grid.astype(x.dtype)))
        x = self.fc0(x)
        x = ms.ops.Transpose()(x, (0, 2, 1))

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

        x_1 = self.conv3(x)
        x_2 = self.w_3(x)
        x = x_1 + x_2

        x = ms.ops.Transpose()(x, (0, 2, 1))
        x = self.fc1(x)
        x = ms.nn.GELU(approximate=False)(x)
        x = self.fc2(x)
        return x
