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

import mindspore
from mindspore import Tensor
from mindelec.common import PI
from scipy import io
import numpy as np
from mindelec.data import Dataset
from pinn.geometry import create_config_from_edict, TimeDomain, GeometryWithTime, Interval
from .sampling_config import *


def get_test_data(test_data_path):
    """load labeled data for evaluation"""
    data = io.loadmat(test_data_path)
    t = data['tt'].flatten()
    x = data['x'].flatten()
    Feature = data['uu']
    Feature_u = np.real(Feature)
    Feature_v = np.imag(Feature)
    features = []
    labels = []
    for i in range(len(x)):
        for j in range(len(t)):
            features.append([x[i], t[j]])
            labels.append([Feature_u[i][j], Feature_v[i][j]])
    features = Tensor(features, dtype=mindspore.float32)
    labels = Tensor(labels, dtype=mindspore.float32)
    return features, labels


def create_random_dataset(config):
    time = TimeDomain("time", config['time_min'], PI/2)
    line = Interval("line", config['coord_min'], config['coord_max'],
                    sampling_config=create_config_from_edict(line_config))
    line_with_time = GeometryWithTime(line, time)
    sampling_config = create_config_from_edict(line_config)
    line_with_time.set_sampling_config(sampling_config)
    geom_dict = {line_with_time: ["domain", "BC", "IC"]}
    dataset = Dataset(geom_dict)

    return dataset


