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

"""
=========================用于检查输入数据的脚本=========================
"""

import sys
import json
import numpy as np

configs = json.load(open("./config.json"))
for filename in configs["train_data_path"]:
    arr = np.load(filename)
    print(arr.shape)
    print("---------------------" + filename + "----------------")
    np.set_printoptions(threshold=sys.maxsize)
    print(arr)
    print(type(arr[0][0]))
