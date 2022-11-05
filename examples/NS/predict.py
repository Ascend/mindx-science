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
from mindspore.common.initializer import XavierUniform
from mindspore import context, Tensor
import mindspore.common.dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from pinn.architecture import MultiScaleFCCell
from src.utils import load_training_data

import matplotlib.pyplot as plt


def l2(lst):
    res = 0
    for i in lst:
        res += float(i) ** 2
    return np.sqrt(res)


def predict(mod, inp):
    tmp = Tensor(inp.reshape(140000, 3), mstype.float32)
    pre = mod(tmp)
    pre = pre.asnumpy()
    return pre

"""evaluation"""

config = json.load(open("./config.json"))

context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target=config["device_target"],
                    device_id=config["device_id"], save_graphs_path="./graph")

# define network
model = MultiScaleFCCell(config["input_size"],
                         config["output_size"],
                         layers=config["layers"],
                         neurons=config["neurons"],
                         input_scale=config["input_scale"],
                         residual=config["residual"],
                         weight_init=XavierUniform(gain=1),
                         act="tanh",
                         num_scales=config["num_scales"],
                         amp_factor=config["amp_factor"],
                         scale_factor=config["scale_factor"]
                         )

# load parameters
param_dict = load_checkpoint(config["load_model_name"])
convert_ckpt_dict = {}
for _, param in model.parameters_and_names():
    convert_name1 = "jac2.model.model.cell_list." + param.name
    CONVERT_NAME2 = "jac2.model.model.cell_list." + ".".join(param.name.split(".")[2:])
    for key in [convert_name1, CONVERT_NAME2]:
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
    uvp_pred = predict(model, xyt_pred)
    x_pred, y_pred, t_pred = xyt_pred[:, 0], xyt_pred[:, 1], xyt_pred[:, 2]
    u_pred, v_pred, p_pred = uvp_pred[:, 0], uvp_pred[:, 1], uvp_pred[:, 2]
    # 筛取t为整数的点
    for ind, element in enumerate(ob_t):
        if ob_t[ind] == t:
            tu.append(ob_u[ind][0])
            tv.append(ob_v[ind][0])
            tp.append(ob_p[ind][0])
            pre_u.append(u_pred[ind])
            pre_v.append(v_pred[ind])
            pre_p.append(p_pred[ind])

    print("测试点数量：")
    print(len(tu))

    print('-----------------------------------------------------------------------------------------------')

    print("u值:t=" + str(t))
    i_ob_u = tu[len(tu) // 2]
    i_pr_u = pre_u[len(pre_u) // 2]
    print(i_ob_u)
    print(i_pr_u)

    for iu, eiu in enumerate(tu):
        uu.append(abs(abs(tu[iu] - pre_u[iu])))
    print(l2(uu) / l2(tu))
    print(sum(uu))
    print(sum(uu) / len(uu))
    uu.clear()
    tu.clear()
    pre_u.clear()

    print('-----------------------------------------------------------------------------------------------')
    print("v值:t=" + str(t))
    i_ob_v = tv[len(tv) // 2]
    i_pr_v = pre_v[len(pre_v) // 2]
    print(i_ob_v)
    print(i_pr_v)

    for iv, eiv in enumerate(tv):
        vv.append(abs(abs(tv[iv] - pre_v[iv])))
    print(l2(vv) / l2(tv))
    print(sum(vv))
    print(sum(vv) / len(vv))
    vv.clear()
    tv.clear()
    pre_v.clear()

    print('-----------------------------------------------------------------------------------------------')
    print("p值:t=" + str(t))
    i_ob_p = tp[len(tp) // 2]
    i_pr_p = pre_p[len(pre_p) // 2]
    axis_p = abs(i_ob_p - i_pr_p)
    print(i_ob_p)
    print(i_pr_p)
    print(axis_p)

    for ip, eip in enumerate(tp):
        pp.append(abs(abs(tp[ip]-pre_p[ip])-axis_p))
    print(l2(pp)/l2(tp))
    print(sum(pp))
    print(sum(pp)/len(pp))
    pp.clear()
    tp.clear()
    pre_p.clear()


