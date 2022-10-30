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
#pylint: disable=W0613
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
#from mindelec.common import MU, EPS, LIGHT_SPEED, PI
from pinn.operators import SecondOrderGrad, Grad


class Possion_equation(Problem):
    r"""
    The 2D Maxwell's equations with 2nd-order Mur absorbed boundary condition.

    Args:
        model (Cell): The solving network.
        config (dict): Setting information.
        domain_name (str): The corresponding column name of data which governed by maxwell's equation.
        bc_name (str): The corresponding column name of data which governed by boundary condition.
        bc_normal (str): The column name of normal direction vector corresponding to specified boundary.
        ic_name (str): The corresponding column name of data which governed by initial condition.
    """
    def __init__(self, model, config, domain_name=None, bc_name=None, bc_normal=None):
        super(Possion_equation, self).__init__()
        self.domain_name = domain_name
        self.bc_name = bc_name
        self.bc_normal = bc_normal
        self.model = model
        self.grad1 = SecondOrderGrad(self.model, 0, 0, 0)
        self.grad2 = SecondOrderGrad(self.model, 1, 1, 0)
        self.grad = Grad(self.model)
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()
        self.mul = ops.Mul()
        self.cast = ops.Cast()
        self.split = ops.Split(1, 2)
        self.concat = ops.Concat(1)
        self.tile = ops.Tile()


        # src space
        self.num_edges = 6 #len(config["vertex_list"])
        self.vertex_list = config["vertex_list"]
        self.coord_min = config["coord_min"]
        self.coord_max = config["coord_max"]
        self.coord_mid = config["coord_mid"]

        '''
        input_scale = config.get("input_scale", [1.0, 1.0, 1.0])
        output_scale = config.get("output_scale", [1.0, 1.0, 1.0])
        self.s_x = Tensor(input_scale[0], mstype.float32)
        self.s_y = Tensor(input_scale[1], mstype.float32)
        self.s_t = Tensor(input_scale[2], mstype.float32)
        self.s_ex = Tensor(output_scale[0], mstype.float32)
        self.s_ey = Tensor(output_scale[1], mstype.float32)
        self.s_hz = Tensor(output_scale[2], mstype.float32)
        '''
    @ms_function
    def governing_equation(self, *output, **kwargs):
        """maxwell equation of TE mode wave"""
        u = output[0]
        data = kwargs[self.domain_name]
        # data.to(mindspore.float16)
        #x = self.reshape(data[:, 0], (-1, 1))
        #y = self.reshape(data[:, 1], (-1, 1))
        du_dxx = self.grad1(data)
        du_dyy = self.grad2(data)
        #du_dx = self.grad(data, 0, 0, u)
        #du_dxx = self.grad(data, 0, 0, du_dx)
        #du_dy = self.grad(data, 1, 0, u)
        #du_dyy = self.grad(data, 1, 0, du_dy)
        return 1 + du_dxx + du_dyy

    @ms_function
    def boundary_condition(self, *output, **kwargs):
        """Dirichlet boundary condition"""
        u = self.cast(output[0], mstype.float32)
        #data = kwargs[self.bc_name]
        '''
        coord_min = self.coord_min
        coord_max = self.coord_max
        coord_mid = self.coord_mid
        data = kwargs[self.bc_name]
        batch_size, _ = data.shape
        bc_all = self.tile(u,(1,self.num_edges))
        attr = ms_np.zeros(shape=(batch_size, self.num_edges))
        attr[:, 0] = ms_np.where(ms_np.isclose(data[:, 0], coord_min[0]), 1.0, 0.0)#(-1,-1)-(-1,1)
        attr[:, 1] = ms_np.where(ms_np.isclose(data[:, 1], coord_max[0]), 1.0, 0.0)#(-1,1)-(0,1)
        attr[:, 2] = ms_np.where(ms_np.isclose(data[:, 0], coord_mid[0])
                                 & ms_np.isclose(data[:, 1], 0.5, atol=0.5, rtol=1e-8),
                                 1.0, 0.0)#(0,1)-(0,0)
        attr[:, 3] = ms_np.where(ms_np.isclose(data[:, 1], coord_mid[0])
                                 & ms_np.isclose(data[:, 0], 0.5, atol=0.5, rtol=1e-8),1.0, 0.0)#(0,0)-(1,0)
        attr[:, 4] = ms_np.where(ms_np.isclose(data[:, 0], coord_max[0]), 1.0, 0.0)#(1,0)-(1,-1)
        attr[:, 5] = ms_np.where(ms_np.isclose(data[:, 1], coord_min[0]), 1.0, 0.0)#(1,-1)-(-1,-1)

        bc_r = self.mul(bc_all, attr)
        return bc_r
        '''
        return u