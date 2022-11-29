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

import json
import numpy as np
import mindspore as ms
from mindelec.architecture import MultiScaleFCCell
from mindspore.common.initializer import XavierUniform
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from base.mindelec_network import SchrodingerNet, dict_transfer_poisson, dict_transfer_ns

configs = json.load(open("./config.json"))

'''create models'''
Poisson_pinn = MultiScaleFCCell(2, 1, layers=5, neurons=50, residual=False,
                                weight_init=XavierUniform(gain=1), input_scale=[1.0, 1.0],
                                act="tanh", num_scales=1, amp_factor=1, scale_factor=1)
Schrodinger_pinn = SchrodingerNet()
NS_pinn = MultiScaleFCCell(3, 3, layers=6, neurons=50, residual=True,
                           weight_init=XavierUniform(gain=1), input_scale=[1.0, 1.0, 1.0],
                           act="tanh", num_scales=1, amp_factor=1, scale_factor=1)

'''load models'''
# Poisson
param_dict_Poisson = load_checkpoint(configs["Poisson_ckpt_path"])
convert_ckpt_dict_Poisson = dict_transfer_poisson(Poisson_pinn, param_dict_Poisson)
load_param_into_net(Poisson_pinn, convert_ckpt_dict_Poisson)

# Schrodinger
param_dict_Schrodinger = load_checkpoint(configs["Schrodinger_ckpt_path"])
load_param_into_net(Schrodinger_pinn, param_dict_Schrodinger)

# NS
param_dict_NS = load_checkpoint(configs["NS_ckpt_path"])
convert_ckpt_dict_NS = dict_transfer_ns(NS_pinn, param_dict_NS)
load_param_into_net(NS_pinn, convert_ckpt_dict_NS)

'''export'''
ms.export(Poisson_pinn, Tensor(np.random.randn(19521, 2), dtype=ms.float32), file_name=configs["Poisson_mid_pinn"],
          file_format=configs["mid_file_format"])
ms.export(Schrodinger_pinn, Tensor(np.random.randn(51456, 2), dtype=ms.float32),
          file_name=configs["Schrodinger_mid_pinn"],
          file_format=configs["mid_file_format"])
ms.export(NS_pinn, Tensor(np.random.randn(355000, 3), dtype=ms.float32), file_name=configs["NS_mid_pinn"],
          file_format=configs["mid_file_format"])
