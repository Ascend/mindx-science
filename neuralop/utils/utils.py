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
import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.callback import Callback
from mindspore.profiler import Profiler


def step_lr_wrapper(decay_per_epoch, scheduler_gamma, ntrain, batch_size,
                    *args, **kwargs):

    decay_per_step = decay_per_epoch * ntrain // batch_size

    def step_lr(learning_rate, cur_step_num):
        if cur_step_num % decay_per_step == 0:
            learning_rate = learning_rate * scheduler_gamma
        return learning_rate

    return step_lr


class MyCallback(Callback):

    def __init__(self, save_per_epoch=None, save_path=None, *args, **kwargs):
        super(MyCallback, self).__init__()
        self.save_per_epoch = save_per_epoch
        self.save_path = save_path

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        learning_rate = cb_params.optimizer.learning_rate
        if isinstance(learning_rate, ms.Tensor) and isinstance(learning_rate.asnumpy(), np.ndarray):
            learning_rate = float(learning_rate.asnumpy())
            print(f'lr: {learning_rate}', flush=True)
        if self.save_per_epoch and self.save_path and (
                cb_params.cur_epoch_num % self.save_per_epoch) == 0:
            ms.save_checkpoint(cb_params.train_network, self.save_path)
            print(
                f'ckpt saved at {self.save_path} after epoch {cb_params.cur_epoch_num}',
                flush=True)

    def end(self, run_context):
        cb_params = run_context.original_args()
        if self.save_per_epoch and self.save_path and (
                cb_params.epoch_num % self.save_per_epoch != 0):
            ms.save_checkpoint(cb_params.train_network, self.save_path)
            print(
                f'ckpt saved at {self.save_path} after epoch #{cb_params.cur_epoch_num}',
                flush=True)


class TimerCallback(Callback):

    def __init__(self):
        super(TimerCallback, self).__init__()
        self.tic, self.toc = None, None

    def begin(self, run_context):
        self.tic = time.perf_counter()

    def end(self, run_context):
        self.toc = time.perf_counter()
        print(f'Timer Duration: {self.toc-self.tic:.6f}')


def load_model(net, path):
    param_dict = load_checkpoint(path)
    load_param_into_net(net, param_dict)
    return net


class MyProfiler():
    def __init__(
            self,
            profile=False,
            profiler_path=None,
            comment=None,
            *args,
            **kwargs):
        self.do_profile = profile
        self.profiler_path = profiler_path if profiler_path else './data'
        self.tic, self.toc = None, None
        self.profiler = None
        print(f'Profiling {comment}...')

    def __enter__(self):
        self.tic = time.perf_counter()
        if self.do_profile:
            self.profiler = Profiler(
                output_path=self.profiler_path,
                start_profile=False)
            self.profiler.start()

    def __exit__(self, *args, **kwargs):
        if self.do_profile:
            self.profiler.stop()
            self.profiler.analyse()
            print(f'profiler data saved to {self.profiler_path}')
        self.toc = time.perf_counter()
        print(f'Duration: {self.toc-self.tic} secs')
