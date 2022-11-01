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
import numpy as np
import scipy
from scipy.io import loadmat

from pinn.data import ExistedDataConfig
from pinn.geometry import Interval, TimeDomain, Geometry, create_config_from_edict, GeometryWithTime, Rectangle

from .sampling_config import src_sampling_config, bc_sampling_config

from pinn.data import SupervisedDataset

def load_training_data(num):
    prefix = "data/"
    data = scipy.io.loadmat(prefix + "cylinder_nektar_wake.mat")
    U_star = data["U_star"]  # N x 2 x T
    P_star = data["p_star"]  # N x T
    t_star = data["t"]  # T x 1
    X_star = data["X_star"]  # N x 2
    N = X_star.shape[0]
    T = t_star.shape[0]
    # Rearrange Data
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T
    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T
    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1
    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    p = PP.flatten()[:, None]  # NT x 1
    # training domain: X × Y = [1, 8] × [−2, 2] and T = [0, 7]
    data1 = np.concatenate([x, y, t, u, v, p], 1)
    data2 = data1[:, :][data1[:, 2] <= 7]
    data3 = data2[:, :][data2[:, 0] >= 1]
    data4 = data3[:, :][data3[:, 0] <= 8]
    data5 = data4[:, :][data4[:, 1] >= -2]
    data_domain = data5[:, :][data5[:, 1] <= 2]
    # choose number of training points: num =7000
    idx = np.random.choice(data_domain.shape[0], num, replace=False)
    x_train = data_domain[idx, 0:1]
    y_train = data_domain[idx, 1:2]
    t_train = data_domain[idx, 2:3]
    u_train = data_domain[idx, 3:4]
    v_train = data_domain[idx, 4:5]
    p_train = data_domain[idx, 5:6]

    res1 = np.hstack((x_train,y_train,t_train))
    res2 = np.hstack((u_train,v_train,p_train))
    return res1,res2


def create_train_dataset(config):
    coord_min = config["coord_min"]
    coord_max = config["coord_max"]
    rectangle = Rectangle("rect", coord_min, coord_max,dtype=np.float32)
    time_interval = TimeDomain("time",config["time_range"][0], config["time_range"][1],dtype=np.float32)
    src_region = GeometryWithTime(rectangle,time_interval)
    src_region.set_name("src")
    src_region.set_sampling_config(create_config_from_edict(src_sampling_config))
    boundary = GeometryWithTime(rectangle,time_interval)
    boundary.set_name("bc")
    boundary.set_sampling_config(create_config_from_edict(bc_sampling_config))

    geom_dict = {src_region:["domain"],
                 #boundary:["BC"]
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

    dataset = SupervisedDataset(geom_dict, existed_data_list=[ob_xyt,ob_uv], label_data={"ob_uv":3})
    return dataset

def test_data_prepare(config):
    return load_training_data(config["test_batch_size"])