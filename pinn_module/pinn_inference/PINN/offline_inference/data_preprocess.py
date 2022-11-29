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

configs = json.load(open("./config.json"))

Poisson_inputs, _ = get_poisson_data(configs["Poisson_data_path"])
np.ascontiguousarray(Poisson_inputs).tofile(configs["Poisson_bin_path"])
Schrodinger_inputs, _ = get_schrodinger_data(configs["Schrodinger_data_path"])
np.ascontiguousarray(Schrodinger_inputs).tofile(configs["Schrodinger_bin_path"])
NS_inputs, _ = get_ns_data(configs["NS_data_path"], 700000)
np.ascontiguousarray(NS_inputs).tofile(configs["NS_bin_path"])
