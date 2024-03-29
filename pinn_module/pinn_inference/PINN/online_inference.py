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
import time
from mindelec.architecture import MultiScaleFCCell
import mindspore.common.dtype as mstype
from mindspore.common.initializer import XavierUniform
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from base.mindelec_network import SchrodingerNet, dict_transfer_poisson, dict_transfer_ns
from base.data_process import get_poisson_data, get_schrodinger_data, get_ns_data
from base.error_calculation import poisson_error, schrodinger_error, ns_error

configs = json.load(open("./config.json"))

# environment
context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="Ascend", save_graphs_path="./graph")

# create models
Poisson_pinn = MultiScaleFCCell(2, 1, layers=5, neurons=50, residual=False,
                                weight_init=XavierUniform(gain=1),
                                act="tanh", num_scales=1, amp_factor=1, scale_factor=1)

Schrodinger_pinn = SchrodingerNet()

NS_pinn = MultiScaleFCCell(3, 3, layers=6, neurons=50, residual=True,
                           weight_init=XavierUniform(gain=1),
                           act="tanh", num_scales=1, amp_factor=1, scale_factor=1)

'''load models'''
# Poisson
param_dict_Poisson = load_checkpoint(configs["Poisson_ckpt_path"])
convert_ckpt_dict_Poisson = dict_transfer_poisson(Poisson_pinn, param_dict_Poisson)
load_param_into_net(Poisson_pinn, convert_ckpt_dict_Poisson)
Poisson_pinn.to_float(mstype.float16)
# Schrodinger
param_dict_Schrodinger = load_checkpoint(configs["Schrodinger_ckpt_path"])
load_param_into_net(Schrodinger_pinn, param_dict_Schrodinger)
Schrodinger_pinn.to_float(mstype.float16)
# NS
param_dict_NS = load_checkpoint(configs["NS_ckpt_path"])
convert_ckpt_dict_NS = dict_transfer_ns(NS_pinn, param_dict_NS)
load_param_into_net(NS_pinn, convert_ckpt_dict_NS)
NS_pinn.to_float(mstype.float16)

'''data preprocess'''
Poisson_inputs, Poisson_labels = get_poisson_data(configs["Poisson_data_path"])
Poisson_inputs = Tensor(Poisson_inputs, mstype.float16)
Schrodinger_inputs, Schrodinger_labels = get_schrodinger_data(configs["Schrodinger_data_path"])
Schrodinger_inputs = Tensor(Schrodinger_inputs, mstype.float16)
NS_inputs, NS_labels = get_ns_data(configs["NS_data_path"], 700000)
NS_inputs = Tensor(NS_inputs.tolist(), mstype.float16)

'''infer'''
tp, ts, tn = [], [], []
for i in range(100):
    T1 = time.time()
    Poisson_results = Poisson_pinn(Poisson_inputs)
    Poisson_time = time.time() - T1
    if i >= 50:
        tp.append(Poisson_time)

for i in range(100):
    T1 = time.time()
    Schrodinger_results = Schrodinger_pinn(Schrodinger_inputs)
    Schrodinger_time = time.time() - T1
    if i >= 50:
        ts.append(Schrodinger_time)

for i in range(100):
    T1 = time.time()
    NS_results = NS_pinn(NS_inputs)
    NS_time = time.time() - T1
    if i >= 50:
        tn.append(NS_time)

'''postprocess'''
Poisson_L2 = poisson_error(Poisson_results.asnumpy(), Poisson_labels)
Schrodinger_L2 = schrodinger_error(Schrodinger_results.asnumpy(), Schrodinger_labels)
u_L2, v_L2 = ns_error(NS_results.asnumpy(), NS_labels)

'''print'''
print("Results of online inference for pinn")
print("Performance")
print("For Poisson: the time is {0}".format(sum(tp) / len(tp)))
print("For Schrodinger: the time is {0}".format(sum(ts) / len(ts)))
print("For NS: the time is {0}".format(sum(tn) / len(tn)))
print("Accuracy")
print("For Poisson: the error is {0}".format(Poisson_L2.item()))
print("For Schrodinger: the error is {0}".format(Schrodinger_L2.item()))
print("For NS: the u/v errors are {0}, {1}".format(u_L2.item(), v_L2.item()))
