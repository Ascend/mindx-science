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
import time
import numpy as np

from mindspore import context
import mindspore.nn as nn
from mindspore.train import DynamicLossScaleManager
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.common.dtype as mstype
from mindspore.common.initializer import XavierUniform
from mindspore.common import set_seed
from mindspore import Tensor, Parameter

from pinn.solver import LossAndTimeMonitor
from pinn.common import L2
from pinn.architecture import MultiScaleFCCell, MTLWeightedLossCell, FCSequential
from pinn.solver import SupervisedSolver
from pinn.loss import SupervisedConstraints
from pinn.common.lr_scheduler import MultiStepLR

from src import create_train_dataset
from src.dataset import test_data_prepare
from src.callback import PredictCallback, GetVariableCallback
from src.ns import NsEquation

set_seed(123456)


def train(config):
    # 动态图or静态图
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True, device_target=config["device_target"],
                        device_id=config["device_id"], save_graphs_path="./graph")

    """创建数据集"""

    elec_train_dataset = create_train_dataset(config)

    train_dataset = elec_train_dataset.create_dataset(batch_size=config["train_batch_size"],
                                                      prebatched_data=True, shuffle=False)

    steps_per_epoch = len(elec_train_dataset)

    """创建模型"""

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

    print("num_losses=", elec_train_dataset.num_dataset)

    mtl = MTLWeightedLossCell(num_losses=elec_train_dataset.num_dataset)

    """创建问题"""
    c1 = Parameter(Tensor(0.0, mstype.float32), name="C1", requires_grad=True)
    c2 = Parameter(Tensor(0.0, mstype.float32), name="C2", requires_grad=True)
    train_prob = {}
    for dataset in elec_train_dataset.all_datasets:
        train_prob[dataset.name] = NsEquation(model=model, c1=c1, c2=c2, config=config,
                                              domain_name=dataset.name + "_points",
                                              bc_name=dataset.name + "_points",
                                              ic_name=dataset.name + "_points")

    train_constraints = SupervisedConstraints(elec_train_dataset, train_prob)

    """优化器"""
    params = model.trainable_params() + mtl.trainable_params() + [c1, c2]
    lr_scheduler = MultiStepLR(config["lr"], config["milestones"],
                               config["lr_gamma"], steps_per_epoch, config["train_epoch"])
    lr = lr_scheduler.get_lr()
    optim = nn.Adam(params, learning_rate=Tensor(lr))

    if config["load_ckpt"]:
        param_dict = load_checkpoint(config["load_ckpt_path"])
        load_param_into_net(model, param_dict)
        load_param_into_net(mtl, param_dict)
    # define solver
    solver = SupervisedSolver(model,
                              optimizer=optim,
                              mode="PINNs",
                              train_constraints=train_constraints,
                              test_constraints=None,
                              metrics={'l2': L2(), 'distance': nn.MAE()},
                              loss_fn=nn.MSELoss(),
                              loss_scale_manager=DynamicLossScaleManager(),
                              amp_level="O0",
                              mtl_weighted_cell=mtl,
                              )

    test_input, test_label = test_data_prepare(config)

    loss_time_callback = LossAndTimeMonitor(steps_per_epoch)
    loss_cb = PredictCallback(model=model, predict_interval=config["predict_interval"],
                              input_data=test_input, label=test_label)
    show_variables = GetVariableCallback(optim=optim, interval=config["show_variables_interval"])
    callbacks = [loss_time_callback, loss_cb, show_variables]
    if config["save_ckpt"]:
        config_ck = CheckpointConfig(save_checkpoint_steps=50,
                                     keep_checkpoint_max=2)
        ckpoint_cb = ModelCheckpoint(prefix='ckpt_NS',
                                     directory=config["save_ckpt_path"], config=config_ck)
        callbacks += [ckpoint_cb]

    solver.train(config["train_epoch"], train_dataset, callbacks=callbacks, dataset_sink_mode=True)


if __name__ == '__main__':
    configs = json.load(open("./config.json"))
    time_beg = time.time()
    train(configs)
    print("End-to-End total time: {} s".format(time.time() - time_beg))
