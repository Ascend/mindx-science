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

from argparse import ArgumentParser

import yaml
import mindspore as ms
from mindspore import nn, Model, context

from data.datasets import Burgers, DarcyFlowFNO, NS2dTime, NSFNO
from architecture.fourier1d import FNO1d
from architecture.fourier2d import FNO2d, FNO2dtime
from architecture.fourier3d import FNO3d
from loss.losses import LPLoss, FNO2dTimeL2Loss
from utils.utils import load_model, MyProfiler

ms.set_seed(1234)

context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')


def test_1d(config, profiler):
    data_base = Burgers(**config['data'])
    test_dataset = data_base.gen_test(**config['data'])
    net = FNO1d(modes=config['model']['modes'])
    load_model(net, config['test']['load_path'])
    loss = LPLoss()
    model = Model(net, loss_fn=loss, metrics={'test mean l2 loss': nn.Loss()})
    model.build(valid_dataset=test_dataset)
    with profiler(comment='Eval FNO1d'):
        ouput = model.eval(test_dataset)
    print(ouput)


def test_2d(config, profiler):
    data_base = DarcyFlowFNO(**config['data'])
    # call train_dataset first to generate data normalizer
    data_base.gen_train(**config['data'])
    test_dataset = data_base.gen_test(**config['data'])
    net = FNO2d(**config['model'])
    load_model(net, config['test']['load_path'])
    loss = LPLoss()
    model = Model(net, loss_fn=loss, metrics={'test mean l2 loss': nn.Loss()})
    model.build(valid_dataset=test_dataset)
    with profiler(comment='Eval FNO2d'):
        ouput = model.eval(test_dataset)
    print(ouput)


def test_2dtime(config, profiler):
    data_base = NS2dTime(**config['data'])
    test_dataset = data_base.gen_test(**config['data'])
    net = FNO2dtime(**config['data'], **config['model'])
    load_model(net, config['test']['load_path'])
    net_with_l2loss = FNO2dTimeL2Loss(net=net,
                                      lploss=LPLoss(),
                                      **config['data'])
    model = Model(net_with_l2loss,
                  loss_fn=None,
                  eval_network=net_with_l2loss,
                  metrics={'test mean L2 loss': nn.Loss()})
    model.build(valid_dataset=test_dataset)
    with profiler(comment='Eval FNO2dtime'):
        ouput = model.eval(test_dataset)
    print(ouput)


def test_3d(config, profiler):
    data_base = NSFNO(**config['data'])
    # call train_dataset first to generate data normalizer
    data_base.gen_train(**config['data'])
    test_dataset = data_base.gen_test(**config['data'])
    net = FNO3d(**config['data'], **config['model'])
    load_model(net, config['test']['load_path'])
    loss = LPLoss()
    model = Model(net, loss_fn=loss, metrics={'test mean l2 loss': nn.Loss()})
    model.build(valid_dataset=test_dataset)
    with profiler(comment='Eval FNO3d'):
        ouput = model.eval(test_dataset)
    print(ouput)


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path',
                        type=str,
                        help='Path to the configuration file')
    options = parser.parse_args()
    config_file = options.config_path
    with open(config_file, 'r', encoding='UTF-8') as stream:
        eval_config = yaml.safe_load(stream)

    assert ('name' in eval_config['model'])
    model_name = eval_config['model']['name']

    if model_name == 'fno1d':
        test_1d(config=eval_config, profiler=MyProfiler)
    elif model_name == 'fno2d':
        test_2d(config=eval_config, profiler=MyProfiler)
    elif model_name == 'fno2dtime':
        test_2dtime(config=eval_config, profiler=MyProfiler)
    elif model_name == 'fno3d':
        test_3d(config=eval_config, profiler=MyProfiler)
    else:
        raise ValueError(
            f'FNO model name config {model_name} is not supported')
