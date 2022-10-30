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
from mindspore.common import *
from mindspore.common.initializer import *
from mindspore import context, Tensor
import mindspore.common.dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from pinn.architecture import MultiScaleFCCell

import matplotlib.pyplot as plt
from scipy.io import loadmat

context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="CPU", save_graphs_path="./graph")

# Load training data
def load_training_data(num):
    data = loadmat("data/cylinder_nektar_wake.mat")
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
    return [x_train, y_train, t_train, u_train, v_train, p_train]

def L2(lst):
    res = 0
    for i in lst:
        res += float(i) ** 2
    return np.sqrt(res)

def predict(model,input):
    tmp = Tensor(input.reshape(140000,3),mstype.float32)
    pred = model(tmp)
    pred = pred.asnumpy()
    return pred

"""evaluation"""

config = json.load(open("./config.json"))

# define network
model = MultiScaleFCCell(config["input_size"], #3
                         config["output_size"], #3
                         layers=config["layers"], #ok
                         neurons=config["neurons"], #ok
                         input_scale=config["input_scale"],# may be changed
                         residual=config["residual"],
                         weight_init=XavierUniform(gain=1),
                         act="tanh",
                         num_scales=config["num_scales"],# may be changed
                         amp_factor=config["amp_factor"],# may be changed
                         scale_factor=config["scale_factor"]# may be changed
                         )

model.to_float(mstype.float16)
model.input_scale.to_float(mstype.float32)

# load parameters
param_dict = load_checkpoint(config["load_model_name"])
convert_ckpt_dict = {}
for _, param in model.parameters_and_names():
    convert_name1 = "jac2.model.model.cell_list." + param.name
    convert_name2 = "jac2.model.model.cell_list." + ".".join(param.name.split(".")[2:])
    for key in [convert_name1, convert_name2]:
        if key in param_dict:
            convert_ckpt_dict[param.name] = param_dict[key]
load_param_into_net(model, convert_ckpt_dict)


uu, vv, pp = [], [], []

tu, tv, tp = [], [], []
pre_u, pre_v, pre_p = [], [], []

# Plot the velocity distribution of the flow field:
for t in range(0, 8):
    [ob_x, ob_y, ob_t, ob_u, ob_v, ob_p] = load_training_data(num=140000)
    xyt_pred = np.hstack((ob_x, ob_y, t * np.ones((len(ob_x), 1))))
    uvp_pred = predict(model,xyt_pred)
    x_pred, y_pred, t_pred = xyt_pred[:, 0], xyt_pred[:, 1], xyt_pred[:, 2]
    u_pred, v_pred, p_pred = uvp_pred[:, 0], uvp_pred[:, 1], uvp_pred[:, 2]
    # tmp_write(u_tru,ob_u[ob_t == t])
    # tmp_write(v_tru,ob_v[ob_t == t])
    # tmp_write(p_tru,ob_p[ob_t == t])
    # 筛取t为整数的点
    for i in range(len(ob_t)):
        if ob_t[i] == t:
            tu.append(ob_u[i][0])
            tv.append(ob_v[i][0])
            tp.append(ob_p[i][0])
            pre_u.append(u_pred[i])
            pre_v.append(v_pred[i])
            pre_p.append(p_pred[i])

    print("测试点数量：")
    print(len(tu))

    print('-----------------------------------------------------------------------------------------------')

    print("u值:t=" + str(t))
    # m_ob_u = sum(ob_u)/len(ob_u)
    # m_pr_u = sum(u_pred)/len(u_pred)
    i_ob_u = tu[len(tu) // 2]
    i_pr_u = pre_u[len(pre_u) // 2]
    # axis_u = abs(i_ob_u - i_pr_u)
    axis_u = 0
    print(i_ob_u)
    print(i_pr_u)
    print(axis_u)

    for i in range(len(tu)):
        uu.append(abs(abs(tu[i] - pre_u[i]) - axis_u))
    print(L2(uu) / L2(tu))
    print(sum(uu))
    print(sum(uu) / len(uu))
    uu.clear()
    tu.clear()
    pre_u.clear()

    # print(len(u_pred))
    # tmp_write(u_pre,u_pred)
    print('-----------------------------------------------------------------------------------------------')
    # m_ob_v = sum(ob_v)/len(ob_v)
    # m_pr_v = sum(v_pred)/len(v_pred)
    print("v值:t=" + str(t))
    i_ob_v = tv[len(tv) // 2]
    i_pr_v = pre_v[len(pre_v) // 2]
    # axis_v = abs(i_ob_v - i_pr_v)
    axis_v = 0
    print(i_ob_v)
    print(i_pr_v)
    print(axis_v)

    for i in range(len(tv)):
        vv.append(abs(abs(tv[i] - pre_v[i]) - axis_v))
    print(L2(vv) / L2(tv))
    print(sum(vv))
    print(sum(vv) / len(vv))
    vv.clear()
    tv.clear()
    pre_v.clear()

    # print(v_pred)
    # tmp_write(v_pre,v_pred)
    print('-----------------------------------------------------------------------------------------------')
    # m_ob_p = sum(ob_p)/len(ob_p)
    # m_pr_p = sum(p_pred)/len(p_pred)
    print("p值:t=" + str(t))
    i_ob_p = tp[len(tp) // 2]
    i_pr_p = pre_p[len(pre_p) // 2]
    axis_p = abs(i_ob_p - i_pr_p)
    print(i_ob_p)
    print(i_pr_p)
    print(axis_p)

    for i in range(len(tp)):
        pp.append(abs(abs(tp[i]-pre_p[i])-axis_p))
    print(L2(pp)/L2(tp))
    print(sum(pp))
    print(sum(pp)/len(pp))
    pp.clear()
    tp.clear()
    pre_p.clear()
    #print(p_pred)
    #tmp_write(p_pre,p_pred)

    x_true = ob_x[ob_t == t]
    y_true = ob_y[ob_t == t]
    u_true = ob_u[ob_t == t]
    v_true = ob_v[ob_t == t]
    p_true = ob_p[ob_t == t]

    prefix = "results/"

    fig, ax = plt.subplots(2, 1)
    cntr0 = ax[0].tricontourf(x_pred, y_pred, u_pred, levels=80, cmap="rainbow")
    cb0 = plt.colorbar(cntr0, ax=ax[0])
    cntr1 = ax[1].tricontourf(x_true, y_true, u_true, levels=80, cmap="rainbow")
    cb1 = plt.colorbar(cntr1, ax=ax[1])
    ax[0].set_title("u-PINN " + "(t=" + str(t) + ")", fontsize=9.5)
    ax[0].axis("scaled")
    ax[0].set_xlabel("X", fontsize=7.5, family="Arial")
    ax[0].set_ylabel("Y", fontsize=7.5, family="Arial")
    ax[1].set_title("u-Reference solution " + "(t=" + str(t) + ")", fontsize=9.5)
    ax[1].axis("scaled")
    ax[1].set_xlabel("X", fontsize=7.5, family="Arial")
    ax[1].set_ylabel("Y", fontsize=7.5, family="Arial")
    fig.tight_layout()
    plt.savefig(prefix + 'u' + '(t=' + str(t) + ').jpg')
    plt.show()

    fig, ax = plt.subplots(2, 1)
    cntr0 = ax[0].tricontourf(x_pred, y_pred, v_pred, levels=80, cmap="rainbow")
    cb0 = plt.colorbar(cntr0, ax=ax[0])
    cntr1 = ax[1].tricontourf(x_true, y_true, v_true, levels=80, cmap="rainbow")
    cb1 = plt.colorbar(cntr1, ax=ax[1])
    ax[0].set_title("v-PINN " + "(t=" + str(t) + ")", fontsize=9.5)
    ax[0].axis("scaled")
    ax[0].set_xlabel("X", fontsize=7.5, family="Arial")
    ax[0].set_ylabel("Y", fontsize=7.5, family="Arial")
    ax[1].set_title("v-Reference solution " + "(t=" + str(t) + ")", fontsize=9.5)
    ax[1].axis("scaled")
    ax[1].set_xlabel("X", fontsize=7.5, family="Arial")
    ax[1].set_ylabel("Y", fontsize=7.5, family="Arial")
    fig.tight_layout()
    plt.savefig(prefix + 'v' + '(t=' + str(t) + ').jpg')
    plt.show()

    fig, ax = plt.subplots(2, 1)
    cntr0 = ax[0].tricontourf(x_pred, y_pred, p_pred, levels=80, cmap="rainbow")
    cb0 = plt.colorbar(cntr0, ax=ax[0])
    cntr1 = ax[1].tricontourf(x_true, y_true, p_true, levels=80, cmap="rainbow")
    cb1 = plt.colorbar(cntr1, ax=ax[1])
    ax[0].set_title("p-PINN " + "(t=" + str(t) + ")", fontsize=9.5)
    ax[0].axis("scaled")
    ax[0].set_xlabel("X", fontsize=7.5, family="Arial")
    ax[0].set_ylabel("Y", fontsize=7.5, family="Arial")
    ax[1].set_title("p-Reference solution " + "(t=" + str(t) + ")", fontsize=9.5)
    ax[1].axis("scaled")
    ax[1].set_xlabel("X", fontsize=7.5, family="Arial")
    ax[1].set_ylabel("Y", fontsize=7.5, family="Arial")
    fig.tight_layout()
    plt.savefig(prefix + 'p' + '(t=' + str(t) + ').jpg')
    plt.show()

