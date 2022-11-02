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
from scipy.io import loadmat


# Load training data
def load_training_data(num):
    data = loadmat("data/cylinder_nektar_wake.mat")
    u_star = data["U_star"]  # N x 2 x T
    p_star = data["p_star"]  # N x T
    t_star = data["t"]  # T x 1
    x_star = data["X_star"]  # N x 2
    n = x_star.shape[0]
    t = t_star.shape[0]
    # Rearrange Data
    xx = np.tile(x_star[:, 0:1], (1, t))  # N x T
    yy = np.tile(x_star[:, 1:2], (1, t))  # N x T
    tt = np.tile(t_star, (1, n)).T  # N x T
    uu = u_star[:, 0, :]  # N x T
    vv = u_star[:, 1, :]  # N x T
    pp = p_star  # N x T
    x = xx.flatten()[:, None]  # NT x 1
    y = yy.flatten()[:, None]  # NT x 1
    tt = tt.flatten()[:, None]  # NT x 1
    u = uu.flatten()[:, None]  # NT x 1
    v = vv.flatten()[:, None]  # NT x 1
    p = pp.flatten()[:, None]  # NT x 1
    # training domain: X × Y = [1, 8] × [−2, 2] and T = [0, 7]
    data1 = np.concatenate([x, y, tt, u, v, p], 1)
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
    return [x_train, y_train, t_train, u_train, v_train, p_train]