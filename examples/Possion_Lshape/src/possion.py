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
# pylint: disable=W0613
"""
2D Possion_equation problem
"""
import mindspore
import mindspore.numpy as ms_np
from mindspore import ms_function
from mindspore import ops
from mindspore import Tensor
import mindspore.common.dtype as mstype

from pinn.solver import Problem
from pinn.operators import SecondOrderGrad, Grad


class PossionEquation(Problem):
    r"""
    The Possion's equations.

    Args:
        model (Cell): The solving network.
        config (dict): Setting information.
        domain_name (str): The corresponding column name of data which governed by maxwell's equation.
        bc_name (str): The corresponding column name of data which governed by boundary condition.
        bc_normal (str): The column name of normal direction vector corresponding to specified boundary.
        ic_name (str): The corresponding column name of data which governed by initial condition.
    """

    def __init__(self, model, config, domain_name=None, bc_name=None, bc_normal=None):
        super(PossionEquation, self).__init__()
        self.domain_name = domain_name
        self.model = model
        self.grad1 = SecondOrderGrad(self.model, 0, 0, 0)
        self.grad2 = SecondOrderGrad(self.model, 1, 1, 0)

    @ms_function
    def governing_equation(self, *output, **kwargs):
        """Possion equation"""
        u = output[0]
        data = kwargs[self.domain_name]
        du_dxx = self.grad1(data)
        du_dyy = self.grad2(data)
        return 1 + du_dxx + du_dyy

    @ms_function
    def boundary_condition(self, *output, **kwargs):
        """Dirichlet boundary condition"""
        return output[0]
