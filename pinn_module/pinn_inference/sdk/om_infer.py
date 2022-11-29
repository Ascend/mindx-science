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
from mindx.sdk.base import Tensor, Model
from base.data_process import get_poisson_data, get_schrodinger_data, get_ns_data
from base.error_calculation import poisson_error, schrodinger_error, ns_error

DEVICE_ID = 0

configs = json.load(open("./config.json"))
Poisson = Model(configs["Poisson_om_path"], DEVICE_ID)
Schrodinger = Model(configs["Schrodinger_om_path"], DEVICE_ID)
NS = Model(configs["NS_om_path"], DEVICE_ID)

'''preprocess'''
# Poisson
Poisson_inputs, Poisson_labels = get_poisson_data(configs["Poisson_data_path"])
Poisson_tensor = Tensor(np.array(Poisson_inputs, dtype=np.float32))
Poisson_tensor.to_device(DEVICE_ID)
Poisson_tensors = [Poisson_tensor]

# Schrodinger
Schrodinger_inputs, Schrodinger_labels = get_schrodinger_data(configs["Schrodinger_data_path"])
Schrodinger_inputs = np.ascontiguousarray(np.array(Schrodinger_inputs, dtype=np.float32))
Schrodinger_tensor = Tensor(Schrodinger_inputs)
Schrodinger_tensor.to_device(DEVICE_ID)
Schrodinger_tensors = [Schrodinger_tensor]

# NS
NS_inputs, NS_labels = get_ns_data(configs["NS_data_path"], 700000)
NS_tensor = Tensor(np.array(NS_inputs, dtype=np.float32))
NS_tensor.to_device(DEVICE_ID)
NS_tensors = [NS_tensor]

'''infer_performance'''
# Poisson
Poisson_outputs = Poisson.infer(Poisson_tensors)
for i in Poisson_outputs:
    i.to_host()
Poisson_results = Poisson_outputs[0]
# Schrodinger
Schrodinger_outputs = Schrodinger.infer(Schrodinger_tensors)
for i in Schrodinger_outputs:
    i.to_host()
Schrodinger_results = Schrodinger_outputs[0]
# NS
NS_outputs = NS.infer(NS_tensors)
for i in NS_outputs:
    i.to_host()
NS_results = NS_outputs[0]

'''postprocess'''
Poisson_L2 = poisson_error(np.array(Poisson_results), Poisson_labels)
Schrodinger_L2 = schrodinger_error(np.array(Schrodinger_results), Schrodinger_labels)
u_L2, v_L2 = ns_error(np.array(NS_results), NS_labels)

'''print'''
print("Results of offline inference for pinn")
print("For Poisson: the error is {0}".format(Poisson_L2.item()))
print("For Schrodinger: the error is {0}".format(Schrodinger_L2.item()))
print("For NS: the u/v errors are {0}, {1}".format(u_L2.item(), v_L2.item()))
