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

from mindspore import nn
from mindelec.architecture.basic_block import LinearBlock as LB
from mindelec.architecture.basic_block import ResBlock as RB
from mindelec.architecture.basic_block import InputScaleNet as IS
from mindelec.architecture.basic_block import FCSequential as FC
from mindelec.architecture.basic_block import MultiScaleFCCell as MSFC

class FCSequential(FC):
    def __init__(self,in_channel,
                 out_channel,
                 layers,
                 neurons,
                 residual=True,
                 act="sin",
                 weight_init='normal',
                 has_bias=True,
                 bias_init='default'):
        super(FCSequential, self).__init__(in_channel, out_channel, layers, neurons, residual, act, weight_init, has_bias, bias_init)

class MultiScaleFCCell(MSFC):
    def __init__(self,in_channel,
                 out_channel,
                 layers,
                 neurons,
                 residual=True,
                 act="sin",
                 weight_init='normal',
                 has_bias=True,
                 bias_init="default",
                 num_scales=4,
                 amp_factor=1.0,
                 scale_factor=2.0,
                 input_scale=None,
                 input_center=None,
                 latent_vector=None):
        super(MultiScaleFCCell, self).__init__(in_channel, out_channel, layers, neurons, residual, act, weight_init, has_bias, bias_init, num_scales, amp_factor, scale_factor, input_scale, input_center, latent_vector)

class Schrodinger_Net(nn.Cell):
    def __init__(self):
        super(Schrodinger_Net, self).__init__()
        self.fc1 = nn.Dense(2, 100, activation='tanh', weight_init='xavier_uniform')
        self.fc2 = nn.Dense(100, 100, activation='tanh', weight_init='xavier_uniform')
        self.fc3 = nn.Dense(100, 100, activation='tanh', weight_init='xavier_uniform')
        self.fc4 = nn.Dense(100, 100, activation='tanh', weight_init='xavier_uniform')
        self.fc5 = nn.Dense(100, 2, weight_init='xavier_uniform')

    def construct(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x
