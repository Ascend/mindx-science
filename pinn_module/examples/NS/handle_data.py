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

import numpy as np
import scipy.io
from src.utils import load_training_data

"===================处理输入数据的脚本==================="


def handle_data(num):
    prefix = "data/"
    [x_train, y_train, t_train, u_train, v_train, p_train] = load_training_data(num)
    res_uv = [x_train, y_train, t_train, u_train, v_train]
    rruv = np.hstack((x_train, y_train, t_train, u_train, v_train)).astype(np.float32)
    rr_out = np.hstack((u_train, v_train, p_train))
    rr1 = np.array(rruv[..., :-2]).astype(np.float32)
    np.save(prefix + 'data_input.npy', rr1)
    np.save(prefix + 'data_uv.npy', rruv)

    return res_uv


handle_data(4096)
