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

import time

import numpy as np

from mindspore.train.callback import Callback
from mindspore import Tensor
import mindspore.common.dtype as mstype


class PredictCallback(Callback):
    """
    Evaluate the model during training.

    Args:
    model (Cell): A testing network.
    predict_ds (Dataset): Dataset to predictuate the model.
    predict_interval (int): Specifies how many epochs to train before prediction.
    input_data (Array): Input test dataset.
    label (Array): Label data.
    batch_size (int): batch size for prediction
    """

    def __init__(self, model, predict_interval, input_data, label, batch_size=8192):
        super(PredictCallback, self).__init__()
        self.model = model
        self.input_data = input_data
        self.label = label
        self.predict_interval = predict_interval
        self.batch_size = min(batch_size, len(input_data))
        self.l2_error = 1.0
        print("check test dataset shape: {}, {}".format(self.input_data.shape, self.label.shape))

    def epoch_end(self, run_context):
        """
        Evaluate the model at the end of epoch.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.predict_interval == 0:
            print("================================Start Evaluation================================")

            test_input_data = self.input_data.reshape(-1, 3)
            label = self.label.reshape(-1, 3)
            index = 0
            prediction = np.zeros(label.shape)
            time_beg = time.time()
            while index < len(test_input_data):
                index_end = min(index + self.batch_size, len(test_input_data))
                test_batch = Tensor(test_input_data[index: index_end, :].tolist(), dtype=mstype.float32)
                predict = self.model(test_batch)
                predict = predict.asnumpy()
                prediction[index: index_end, :] = predict[:, :]
                index = index_end
            print("Total prediction time: {} s".format(time.time() - time_beg))
            error = label[:, :2] - prediction[:, :2]
            l2_error = np.sqrt(np.sum(np.square(error[:, 0]))) / np.sqrt(np.sum(np.square(label[:, 0])))
            print("l2_error(u,v): ", l2_error)
            print("=================================End Evaluation=================================")
            self.l2_error = l2_error

    def get_l2_error(self):
        return self.l2_error


class GetVariableCallback(Callback):

    def __init__(self, optim, interval):
        super(GetVariableCallback, self).__init__()
        self.optim = optim
        self.interval = interval

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.interval == 0:
            print("======================Show Variables=======================")
            for _, param in self.optim.parameters_and_names():
                if param.name == "C1" or param.name == "C2":
                    print(Tensor(param))
            print("======================Show Variables=======================")


