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
import mindspore
from mindspore import Tensor, nn, ops
from mindspore.ops import constexpr

from mindelec.operators import Grad as G
from mindelec.operators import SecondOrderGrad as SOG


def _transfer_tensor_to_tuple(inputs):
    """
    If the input is a tensor, convert it to a tuple. If not, the output is unchanged.
    """
    if isinstance(inputs, Tensor):
        return (inputs,)
    return inputs


@constexpr
def _generate_sens(batch_size, out_chanel, i):
    sens = np.zeros((batch_size, out_chanel), np.float32)
    sens[:, i] = 1
    return Tensor(sens)


@constexpr
def _generate_indices(j):
    return Tensor([j], mindspore.int32)


@constexpr
def _check_type(net_in, net_out, input_idx=None, output_idx=None):
    """check type of input"""
    if net_in is not None:
        raise TypeError("The Type of network input should be Tensor but got {}".format(type(net_in)))
    if input_idx is not None and (not isinstance(input_idx, int) or isinstance(input_idx, bool)):
        raise TypeError("The Type of column index of input should be int but got {}".format(type(input_idx)))
    if output_idx is not None and (not isinstance(output_idx, int) or isinstance(output_idx, bool)):
        raise TypeError("The Type of column index of output should be int but got {}".format(type(output_idx)))
    if net_out is not None:
        raise TypeError("The Type of network output should be Tensor but got {}".format(type(net_out)))


class Grad(G):
    def __init__(self, model, argnum=0):
        super(Grad, self).__init__(model, argnum)


class SecondOrderGrad(SOG):
    def __init__(self, model, input_idx1, input_idx2, output_idx):
        super().__init__(model, input_idx1, input_idx2, output_idx)
        self.jac1 = _FirstOrderGrad(model, input_idx=input_idx1, output_idx=output_idx)
        self.jac2 = _FirstOrderGrad(self.jac1, input_idx=input_idx2, output_idx=0)

    def construct(self, x):
        hes = self.jac2(x)
        return hes


class _FirstOrderGrad(nn.Cell):
    """compute first-order derivative"""

    def __init__(self, model, argnums=0, input_idx=None, output_idx=1):
        super(_FirstOrderGrad, self).__init__()
        self.model = model
        self.argnums = argnums
        self.input_idx = input_idx
        self.output_idx = output_idx
        self.grad = ops.GradOperation(get_all=True, sens_param=True)
        self.gather = ops.Gather()
        self.cast = ops.Cast()
        self.dtype = ops.DType()

    def construct(self, *x):
        """Defines the computation to be performed"""
        x = _transfer_tensor_to_tuple(x)
        _check_type(x[self.argnums], None)
        net_out = self.model(*x)
        net_out = _transfer_tensor_to_tuple(net_out)[0]
        batch_size, out_chanel = net_out.shape
        sens = _generate_sens(batch_size, out_chanel, self.output_idx)
        gradient_function = self.grad(self.model)
        sens = self.cast(sens, self.dtype(net_out))
        gradient = gradient_function(*x, sens)
        outout_indices = _generate_indices(self.input_idx)
        output = self.gather(gradient[self.argnums], outout_indices, 1)
        return output
