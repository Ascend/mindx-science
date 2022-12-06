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
from mindspore.train.callback import Callback
from mindspore import ops

import mindspore


class TlossCallback(Callback):
    def __init__(self, net, feature, label):
        super(TlossCallback, self).__init__()
        self.net = net
        self.feature = feature
        self.label = label
        self.rs = ops.ReduceSum(keep_dims=True)

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        predict = self.net(self.feature)
        result = ops.L2Loss()(predict - self.label)
        predict_h = ops.Sqrt()(self.rs(ops.Pow()(predict, 2), 1))
        label_h = ops.Sqrt()(self.rs(ops.Pow()(self.label, 2), 1))
        print('test_loss:', result)
        print('u, v=', ops.Sqrt()(
            self.rs(ops.Pow()(predict - self.label, 2), 0) /
            self.rs(ops.Pow()(self.label, 2), 0))
              )
        h = ops.Sqrt()(
            self.rs(ops.Pow()(predict_h - label_h, 2), 0) /
            self.rs(ops.Pow()(label_h, 2), 0))
        print('h=', h)
        if h[0][0] < 0.00172:
            file_name = "epoch:" + str(cb_params.cur_epoch_num) + "_result" + ".ckpt"
            mindspore.save_checkpoint(save_obj=self.net, ckpt_file_name=file_name)
            run_context.request_stop()
