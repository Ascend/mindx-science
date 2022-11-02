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
import numpy as np
from pinn.operators import SecondOrderGrad, Grad
from mindspore import ops, Tensor, ms_function
from pinn.solver import Problem


class Schrodinger(Problem):
    def __init__(self, model, domain_name=None, bc_name=None, ic_name=None):
        super(Schrodinger, self).__init__()
        self.domain_name = domain_name
        self.bc_name = bc_name
        self.ic_name = ic_name
        self.model = model
        # operators
        self.u_xx = SecondOrderGrad(self.model, 0, 0, 0)
        self.v_xx = SecondOrderGrad(self.model, 0, 0, 1)
        self.grad = Grad(model)
        self.pow = ops.Pow()
        self.rs_t = ops.ReduceSum(keep_dims=True)
        self.split = ops.Split(1, 2)
        self.zero = ops.ZerosLike()
        self.cosh = ops.Cosh()
        self.concat = ops.Concat(1)
        self.rs = ops.ReduceSum()

    @ms_function
    def governing_equation(self, *output, **kwargs):
        data = kwargs[self.domain_name]
        h = output[0]

        du_xx = self.u_xx(data)
        dv_xx = self.v_xx(data)
        du_t = self.grad(data, 1, 0, h)
        dv_t = self.grad(data, 1, 1, h)
        h_2 = self.rs_t(self.pow(h, 2), 1)
        h2_s = self.split(h_2 * h)
        h2_u = h2_s[0]
        h2_v = h2_s[1]
        r = self.concat((-dv_t + 0.5 * du_xx + h2_u, du_t + 0.5 * dv_xx + h2_v))
        return r

    @ms_function
    def boundary_condition(self, *output, **kwargs):
        h = output[0]
        data = kwargs[self.bc_name]
        data1 = data * Tensor(np.array([-1, 1]), mindspore.float32)
        h1 = self.model(data1)
        du_x = self.grad(data, 0, 0, h)
        dv_x = self.grad(data, 0, 1, h)
        du1_x = self.grad(data1, 0, 0, h1)
        dv1_x = self.grad(data1, 0, 1, h1)
        dhx = self.concat((du_x, dv_x))
        dh1x = self.concat((du1_x, dv1_x))
        return ops.Sqrt()(self.pow(h - h1, 2) + self.pow(dhx - dh1x, 2))

    @ms_function
    def initial_condition(self, *output, **kwargs):
        h = output[0]
        data = kwargs[self.ic_name]
        sechic = 1 / self.cosh(data)
        sechx = self.split(sechic)[0]
        error0 = self.concat((2 * sechx, self.zero(sechx)))
        return h - error0
