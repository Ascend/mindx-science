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

import mindspore
from mindspore import ms_function
from mindspore import ops, Tensor

from pinn.solver import Problem
from pinn.operators import SecondOrderGrad, Grad


class NsEquation(Problem):
    def __init__(self, model, c1, c2, config, domain_name=None, bc_name=None, ic_name=None):
        super(NsEquation, self).__init__()
        self.domain_name = domain_name
        self.bc_name = bc_name
        self.ic_name = ic_name
        self.model = model
        self.grad = Grad(self.model)
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()
        self.split = ops.Split(1, 2)
        self.concat = ops.Concat(1)
        self.secondgrad1 = SecondOrderGrad(self.model, 0, 0, 0)
        self.secondgrad2 = SecondOrderGrad(self.model, 1, 1, 0)
        self.secondgrad3 = SecondOrderGrad(self.model, 0, 0, 1)
        self.secondgrad4 = SecondOrderGrad(self.model, 1, 1, 1)
        self.c1 = c1
        self.c2 = c2

    @ms_function
    def governing_equation(self, *output, **kwargs):
        result = output[0]
        u = self.reshape(result[:, 0], (-1, 1))
        v = self.reshape(result[:, 1], (-1, 1))
        data = kwargs.get(self.domain_name)
        du_x = self.grad(data, 0, 0, result)
        du_y = self.grad(data, 1, 0, result)
        du_t = self.grad(data, 2, 0, result)
        dv_x = self.grad(data, 0, 1, result)
        dv_y = self.grad(data, 1, 1, result)
        dv_t = self.grad(data, 2, 1, result)
        dp_x = self.grad(data, 0, 2, result)
        dp_y = self.grad(data, 1, 2, result)
        du_xx = self.secondgrad1(data)
        du_yy = self.secondgrad2(data)
        dv_xx = self.secondgrad3(data)
        dv_yy = self.secondgrad4(data)
        x_momentum = du_t + self.c1 * (u * du_x + v * du_y) + dp_x - self.c2 * (du_xx + du_yy)
        y_momentum = dv_t + self.c1 * (u * dv_x + v * dv_y) + dp_y - self.c2 * (dv_xx + dv_yy)
        return ops.Concat(1)((x_momentum, y_momentum))

