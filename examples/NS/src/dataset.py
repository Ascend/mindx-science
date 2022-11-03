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
from pinn.data import ExistedDataConfig
from pinn.geometry import Interval, TimeDomain, Geometry, create_config_from_edict, GeometryWithTime, Rectangle
from pinn.data import SupervisedDataset
from .sampling_config import src_sampling_config, bc_sampling_config
from .utils import load_training_data


def load_data(num):
    prefix = "data/"
    [x_train, y_train, t_train, u_train, v_train, p_train] = load_training_data(num)
    res1 = np.hstack((x_train, y_train, t_train))
    res2 = np.hstack((u_train, v_train, p_train))
    return res1, res2


def create_train_dataset(config):
    coord_min = config["coord_min"]
    coord_max = config["coord_max"]
    rectangle = Rectangle("rect", coord_min, coord_max, dtype=np.float32)
    time_interval = TimeDomain("time", config["time_range"][0], config["time_range"][1], dtype=np.float32)
    src_region = GeometryWithTime(rectangle, time_interval)
    src_region.set_name("src")
    src_region.set_sampling_config(create_config_from_edict(src_sampling_config))

    geom_dict = {src_region: ["domain"],
                 }

    ob_xyt = ExistedDataConfig(name='ob_xyt',
                               data_dir=[config["train_data_path"][0]],
                               columns_list=['points'],
                               data_format="npy",
                               constraint_type="Equation")

    "-----------------------------------------------------------------------"
    "以下是有监督数据，[x,y,t,u/v/p] 四维，修改了mindelec中的nei_with_loss对label进行特殊处理来计算有监督数据的损失"

    ob_uv = ExistedDataConfig(name='ob_uv',
                              data_dir=[config["train_data_path"][1]],
                              columns_list=['points'],
                              data_format="npy",
                              constraint_type="Equation")

    dataset = SupervisedDataset(geom_dict, existed_data_list=[ob_xyt, ob_uv], label_data={"ob_uv": 3})
    return dataset


def test_data_prepare(config):
    return load_data(config["test_batch_size"])
