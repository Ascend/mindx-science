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


class SchrodingerNet(nn.Cell):
    def __init__(self):
        super(SchrodingerNet, self).__init__()
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


def dict_transfer_poisson(model, param_dict):
    convert_ckpt_dict = {}
    for _, param in model.parameters_and_names():
        convert_name1 = "jac2.model.model.cell_list." + param.name
        convert_name2 = "jac2.model.model.cell_list." + ".".join(param.name.split(".")[2:])
        for key in [convert_name1, convert_name2]:
            if key in param_dict:
                convert_ckpt_dict[param.name] = param_dict[key]
    return convert_ckpt_dict


def dict_transfer_ns(model, param_dict):
    convert_ckpt_dict = {}
    for _, param in model.parameters_and_names():
        convert_name1 = "jac2.model.model.cell_list." + param.name
        convert_name2 = "jac2.model.model.cell_list." + ".".join(param.name.split(".")[2:])
        for key in [convert_name1, convert_name2]:
            if key in param_dict:
                convert_ckpt_dict[param.name] = param_dict[key]
    return convert_ckpt_dict
