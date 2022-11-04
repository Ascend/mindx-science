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

import os
import json
import time
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from mindspore import context, Tensor, ops, nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net

import src.dataset

from pinn.architecture import SchrodingerNet
sys.path.append(str(Path(__file__).resolve().parents[2]))



context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="Ascend", save_graphs_path="./graph")


def evaluation(config):
    """evaluation"""

    # define network
    model = SchrodingerNet()

    # load parameters
    param_dict = load_checkpoint(config["load_ckpt_path"])
    load_param_into_net(model, param_dict)

    # load test dataset
    inputs, label = src.dataset.get_test_data(config["test_data_path"])
    time_beg = time.time()
    predict = model(inputs)
    print('example for 8000 to 8050:\n', np.concatenate((predict[8000:8050].asnumpy()
                                                         , label[8000:8050].asnumpy()), axis=1))
    print("predict total time: {} s".format(time.time() - time_beg))

    # get accuracy
    rs = ops.ReduceSum(keep_dims=True)
    print('u, v=', ops.Sqrt()(
        rs(ops.Pow()(predict - label, 2), 0) /
        rs(ops.Pow()(label, 2), 0))
          )
    predict_h = ops.Sqrt()(rs(ops.Pow()(predict, 2), 1))
    label_h = ops.Sqrt()(rs(ops.Pow()(label, 2), 1))
    print('h=', ops.Sqrt()(
        rs(ops.Pow()(predict_h - label_h, 2), 0) /
        rs(ops.Pow()(label_h, 2), 0))
          )

    # draw picture
    x_lower = -5
    x_upper = 5
    t_lower = 0
    t_upper = np.pi / 2

    fig, axes = plt.subplots(3, 2, figsize=(150, 60), squeeze=False)

    # Plot label

    axes[0, 0].set_title("Results")
    axes[0, 0].set_ylabel("Real part")
    axes[0, 0].imshow(
        label[:, 0].reshape(256, 201).asnumpy(),
        interpolation="nearest",
        cmap="viridis",
        extent=[t_lower, t_upper, x_lower, x_upper],
        origin="lower",
        aspect="auto",
    )
    axes[1, 0].set_ylabel("Imaginary part")
    axes[1, 0].imshow(
        label[:, 1].reshape(256, 201).asnumpy(),
        interpolation="nearest",
        cmap="viridis",
        extent=[t_lower, t_upper, x_lower, x_upper],
        origin="lower",
        aspect="auto",
    )
    axes[2, 0].set_ylabel("Amplitude")
    axes[2, 0].imshow(
        label_h.reshape(256, 201).asnumpy(),
        interpolation="nearest",
        cmap="viridis",
        extent=[t_lower, t_upper, x_lower, x_upper],
        origin="lower",
        aspect="auto",
    )

    # Plot predictions

    axes[0, 1].set_title("Results_label")
    axes[0, 1].set_ylabel("Real part")
    axes[0, 1].imshow(
        predict[:, 0].reshape(256, 201).asnumpy(),
        interpolation="nearest",
        cmap="viridis",
        extent=[t_lower, t_upper, x_lower, x_upper],
        origin="lower",
        aspect="auto",
    )
    axes[1, 1].set_ylabel("Imaginary part_label")
    axes[1, 1].imshow(
        predict[:, 1].reshape(256, 201).asnumpy(),
        interpolation="nearest",
        cmap="viridis",
        extent=[t_lower, t_upper, x_lower, x_upper],
        origin="lower",
        aspect="auto",
    )
    axes[2, 1].set_ylabel("Amplitude_label")
    axes[2, 1].imshow(
        predict_h.reshape(256, 201).asnumpy(),
        interpolation="nearest",
        cmap="viridis",
        extent=[t_lower, t_upper, x_lower, x_upper],
        origin="lower",
        aspect="auto",
    )

    # 保存图片
    plt.show()
    fig.savefig("eval/result" + ".png")


if __name__ == '__main__':
    print("pid:", os.getpid())
    configs = json.load(open("./config.json"))
    print("check config: {}".format(configs))
    evaluation(configs)
