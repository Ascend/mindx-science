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

from scipy import io
import numpy as np

"""data preprocess"""


def get_schrodinger_data(test_data_path):
    """load labeled data for evaluation"""
    data = io.loadmat(test_data_path)
    t = data['tt'].flatten()
    x = data['x'].flatten()
    feature = data['uu']
    feature_u = np.real(feature)
    feature_v = np.imag(feature)
    features = []
    labels = []
    for i, ei in enumerate(x):
        for j, ej in enumerate(t):
            features.append([x[i], t[j]])
            labels.append([feature_u[i][j], feature_v[i][j]])
    features = np.array(features)
    labels = np.array(labels)
    return features, labels


def get_poisson_data(test_data_path):
    """load labeled data for evaluation"""
    data = np.load(test_data_path)
    inputs = data['X_test']
    labels = data['y_ref']
    return inputs, labels


def get_ns_data(test_data_path, num):
    """put the labeled data in order"""
    [x_train, y_train, t_train, u_train, v_train, p_train] = load_ns_data(test_data_path, num)
    res1 = np.hstack((x_train, y_train, t_train))
    res2 = np.hstack((u_train, v_train, p_train))
    return res1, res2


def load_ns_data(test_data_path, num):
    """load labeled data for evaluation"""
    data = io.loadmat(test_data_path)
    u_s, p_s, t_s, x_s = data["U_star"], data["p_star"], data["t"], data["X_star"]
    n, t = x_s.shape[0], t_s.shape[0]
    xx, yy = np.tile(x_s[:, 0:1], (1, t)), np.tile(x_s[:, 1:2], (1, t))
    tt = np.tile(t_s, (1, n)).T
    uu, vv = u_s[:, 0, :], u_s[:, 1, :]
    pp = p_s

    tmp = [xx.flatten()[:, None], yy.flatten()[:, None], tt.flatten()[:, None], uu.flatten()[:, None],
           vv.flatten()[:, None], pp.flatten()[:, None]]
    data1 = np.concatenate(tmp, 1)
    data2 = data1[:, :][data1[:, 2] <= 7]

    data3 = data2[:, :][data2[:, 0] >= 1]
    data4 = data3[:, :][data3[:, 0] <= 8]
    data5 = data4[:, :][data4[:, 1] >= -2]
    data_domain = data5[:, :][data5[:, 1] <= 2]
    if (num < data_domain.shape[0]):
        idx = np.random.choice(data_domain.shape[0], num, replace=False)
        return [data_domain[idx, 0:1], data_domain[idx, 1:2], data_domain[idx, 2:3], data_domain[idx, 3:4],
                data_domain[idx, 4:5], data_domain[idx, 5:6]]
    else:
        return [data_domain[:, 0:1], data_domain[:, 1:2], data_domain[:, 2:3], data_domain[:, 3:4],
                data_domain[:, 4:5], data_domain[:, 5:6]]
