# Copyright 2021 Huawei Technologies Co., Ltd
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

# import sys
# sys.path.append(r"F:\\0-code\\pinnwor")
# print(sys.path)

"""train process"""
import os
import json
import math
import time
import numpy as np

from mindspore.common import *
from mindspore.common.initializer import *
from mindspore import context, Tensor, nn
from mindspore.train import DynamicLossScaleManager
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.common.dtype as mstype
from mindspore.common.initializer import HeUniform

from pinn.loss import Constraints
from pinn.solver import Solver, LossAndTimeMonitor
from pinn.common import L2
from pinn.architecture import MultiScaleFCCell, MTLWeightedLossCell, FCSequential

from src import get_test_data, create_random_dataset
from src.Possion import Possion_equation
from src import MultiStepLR, PredictCallback
from src import visual_result

# 是否需要修改
set_seed(123456)
np.random.seed(123456)


def train(config):
    # Static Graph
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True, device_target=config["device_target"],
                        device_id=config["device_id"], save_graphs_path="./graph")

    """training process"""
    # dataset
    elec_train_dataset = create_random_dataset(config)
    train_dataset = elec_train_dataset.create_dataset(batch_size=config["train_batch_size"],
                                                      shuffle=True,
                                                      prebatched_data=True, #改了
                                                      drop_remainder=True)

    steps_per_epoch = len(elec_train_dataset)
    print("check train dataset size: ", len(elec_train_dataset))

    # define network

    model = MultiScaleFCCell(config["input_size"], #2
                             config["output_size"], #1
                             layers=config["layers"], #ok
                             neurons=config["neurons"], #ok
                             input_scale=config["input_scale"],# may be changed
                             residual=config["residual"],#与示例一致
                             weight_init=XavierUniform(gain=1),#与示例一致
                             act="tanh",#与示例一致
                             num_scales=config["num_scales"],# may be changed
                             amp_factor=config["amp_factor"],# may be changed
                             scale_factor=config["scale_factor"]# may be changed
                             )
    # model.to_float(mstype.float16)
    # model.input_scale.to_float(mstype.float32)
    '''
    model = FCSequential(in_channel=config["input_size"],
                         out_channel=config["output_size"],
                         layers=config["layers"],
                         neurons=config["neurons"],
                         residual=config["residual"],
                         act="tanh",
                         weight_init=XavierUniform(gain=1))
    model.to_float(mstype.float32)
    '''

    print("num_losses=", elec_train_dataset.num_dataset)
    mtl = MTLWeightedLossCell(num_losses=elec_train_dataset.num_dataset) # 2 losses

    # define problem
    train_prob = {}
    for dataset in elec_train_dataset.all_datasets:
        print(dataset)
        train_prob[dataset.name] = Possion_equation(model=model, config=config,
                                                domain_name=dataset.name + "_points",
                                                bc_name=dataset.name + "_points")
    print("check problem: ", train_prob)
    train_constraints = Constraints(elec_train_dataset, train_prob)

    # optimizer
    params = model.trainable_params() + mtl.trainable_params()
    optim = nn.Adam(params, learning_rate=Tensor(config["lr"]))

    if config["load_ckpt"]:
        param_dict = load_checkpoint(config["load_ckpt_path"])
        load_param_into_net(model, param_dict)
        load_param_into_net(mtl, param_dict)

    # define solver
    # not quite sure
    solver = Solver(model,
                    optimizer=optim,
                    mode="PINNs",
                    train_constraints=train_constraints,
                    test_constraints=None,
                    metrics={'l2': L2(), 'distance': nn.MSE()},
                    loss_fn=nn.MSELoss(),
                    loss_scale_manager=DynamicLossScaleManager(),
                    mtl_weighted_cell=mtl,
                    amp_level="O3"
                    )
    print("steps_per_epoch=", steps_per_epoch)
    loss_time_callback = LossAndTimeMonitor(steps_per_epoch)
    callbacks = [loss_time_callback]
    if config.get("train_with_eval", False):
        inputs, label = get_test_data(config["test_data_path"])
        predict_callback = PredictCallback(model, inputs, label, config=config) #, visual_fn=visual_result)
        callbacks += [predict_callback]
    if config["save_ckpt"]:
        config_ck = CheckpointConfig(save_checkpoint_steps=10,
                                     keep_checkpoint_max=2)
        ckpoint_cb = ModelCheckpoint(prefix='ckpt_possion',
                                     directory=config["save_ckpt_path"], config=config_ck)
        callbacks += [ckpoint_cb]
    print("callbacks=", callbacks)
    solver.train(config["train_epoch"], train_dataset, callbacks=callbacks, dataset_sink_mode=True)


if __name__ == '__main__':
    print("pid:", os.getpid())
    configs = json.load(open("./config.json"))
    print("check config: {}".format(configs))
    time_beg = time.time()
    train(configs)
    print("End-to-End total time: {} s".format(time.time() - time_beg))
