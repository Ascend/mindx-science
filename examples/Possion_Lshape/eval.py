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
"""evaluation process"""
import os
import json
import time
import copy
import numpy as np
from mindspore.common import *
from mindspore.common.initializer import *
from mindspore import context, Tensor
import mindspore.common.dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from pinn.architecture import MultiScaleFCCell

from src import get_test_data
from src import visual_result
from src.utils import cloud_picture


context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="Ascend", save_graphs_path="./graph")

def evaluation(config):
    """evaluation"""
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

    model.to_float(mstype.float16)
    model.input_scale.to_float(mstype.float32)

    # load parameters
    param_dict = load_checkpoint(config["load_model_name"])
    convert_ckpt_dict = {}
    for _, param in model.parameters_and_names():
        convert_name1 = "model.cell_list." + param.name
        convert_name2 = "model.cell_list." + ".".join(param.name.split(".")[2:])
        for key in [convert_name1, convert_name2]:
            if key in param_dict:
                convert_ckpt_dict[param.name] = param_dict[key]
    load_param_into_net(model, convert_ckpt_dict)

    # load test dataset
    inputs, label = get_test_data(config["test_data_path"])
    outputs_size = config.get("outputs_size", 1)
    inputs_size = config.get("inputs_size", 2)
    outputs_scale = np.array(config["output_scale"], dtype=np.float32)
    batch_size = config.get("test_batch_size", 8192)

    inputs_each = [inputs]

    index = 0
    prediction_each = np.zeros(label.shape)
    prediction_each = prediction_each.reshape((-1, outputs_size))
    time_beg = time.time()
    while index < len(inputs_each[0]):
        index_end = min(index + batch_size, len(inputs_each[0]))
        # predict each physical quantity respectively in order to keep consistent with fdtd on staggered mesh
        for i in range(outputs_size):
            test_batch = Tensor(inputs_each[i][index: index_end, :], mstype.float32)
            predict = model(test_batch)
            predict = predict.asnumpy()
            prediction_each[index: index_end, i] = predict[:, i] * outputs_scale[i]
        index = index_end

    print("predict total time: {} s".format(time.time() - time_beg))
    prediction = prediction_each.reshape(label.shape)
    vision_path = config.get("vision_path", "./vision")
    #visual_result(inputs, label, prediction, path=vision_path, name="predict")
    cloud_picture(inputs, label, prediction, path=vision_path)

    # get accuracy
    error = label - prediction
    print("label=", label)
    print("prediction=", prediction)
    l2_error_u = np.sqrt(np.sum(np.square(error))) / np.sqrt(np.sum(np.square(label)))
    print("l2_error, u: ", l2_error_u)

if __name__ == '__main__':
    print("pid:", os.getpid())
    configs = json.load(open("./config.json"))
    print("check config: {}".format(configs))
    evaluation(configs)
