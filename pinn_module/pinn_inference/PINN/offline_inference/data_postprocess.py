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
from base.data_process import get_poisson_data, get_schrodinger_data, get_ns_data
from base.error_calculation import poisson_error, schrodinger_error, ns_error

configs = json.load(open("./config.json"))
Poisson_results = np.loadtxt(configs["Poisson_result_path"])
Schrodinger_results = np.loadtxt(configs["Schrodinger_result_path"])
NS_results = np.loadtxt(configs["NS_result_path"])

_, Poisson_labels = get_poisson_data(configs["Poisson_data_path"])
_, Schrodinger_labels = get_schrodinger_data(configs["Schrodinger_data_path"])
_, NS_labels = get_ns_data(configs["NS_data_path"], 700000)

Poisson_L2 = poisson_error(Poisson_results.reshape(-1, 1), Poisson_labels)
Schrodinger_L2 = schrodinger_error(Schrodinger_results, Schrodinger_labels)
u_L2, v_L2 = ns_error(NS_results, NS_labels)

print("Poisson error=", Poisson_L2)
print("Schrodinger error=", Schrodinger_L2)
print("NS error: u={0}, v={1}".format(u_L2, v_L2))
