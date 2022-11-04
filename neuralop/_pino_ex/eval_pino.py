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

from data.datasets import DarcyFlowPINO, NSPINO, get_forcing
from architecture.fourier2d import FNN2d
from architecture.fourier3d import FNN3d
from loss.losses import (DarcyLoss, DarcyDataLoss, DarcyEqnLoss,
                         LPLoss, NSTestDataLoss, NSTestEqnLoss, PINOLoss3D)
from utils.utils import load_model, MyProfiler


ms.set_seed(1234)

context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')


def test_darcy(config, profiler):
    data_base = DarcyFlowPINO(**config['data'])
    test_dataset1 = data_base.gen_test(**config['data'])
    test_dataset2 = data_base.gen_test(**config['data'])
    mollifier = data_base.mollifier(**config['data'])
    net = FNN2d(**config['model'])
    load_model(net, config['test']['load_path'])
    net_with_dataloss = DarcyDataLoss(net=net,
                                      mollifier=mollifier,
                                      lploss=LPLoss(),
                                      **config['data'])
    model1 = Model(net_with_dataloss,
                   loss_fn=None,
                   eval_network=net_with_dataloss,
                   metrics={'test mean data L2 loss': nn.Loss()})

    net_with_eqnloss = DarcyEqnLoss(net=net,
                                    mollifier=mollifier,
                                    darcyloss=DarcyLoss(),
                                    **config['data'])
    model2 = Model(net_with_eqnloss,
                   loss_fn=None,
                   eval_network=net_with_eqnloss,
                   metrics={'test mean eqn L2 loss': nn.Loss()})

    model1.build(valid_dataset=test_dataset1)
    model2.build(valid_dataset=test_dataset2)
    with profiler():
        output1 = model1.eval(test_dataset1)
        output2 = model2.eval(test_dataset2)
    print(output1)
    print(output2)


def test_3d(config, profiler):
    data_s = config['data']['nx_data']
    re = config['data']['Re']
    batch_size = config['data']['batch_size']
    test_path = config['data']['test_path']
    ntest = config['data']['ntest']
    time_interval = config['data']['time_interval']
    shuffle = config['data']['shuffle']

    v = float(1 / re)
    test_forcing = get_forcing(data_s)
    test_base = NSPINO(datapath=test_path, **config['data'])
    test_base.append_mesh(nsample=ntest, train=False)
    test_dataset1 = test_base.gen_dataset(
        batch_size=batch_size, shuffle=shuffle)
    test_dataset2 = test_base.gen_dataset(
        batch_size=batch_size, shuffle=shuffle)

    net = FNN3d(**config['model'])
    load_model(net, config['test']['load_path'])

    net_test_loss_data = NSTestDataLoss(net=net,
                                         lploss=LPLoss(),
                                         **config['data'])

    model1 = Model(net_test_loss_data,
                   loss_fn=None,
                   eval_network=net_test_loss_data,
                   metrics={'test mean l2 loss': nn.Loss()})

    net_test_loss_eqn = NSTestEqnLoss(net=net,
                                       nsloss=PINOLoss3D(
                                           test_forcing, v, time_interval),
                                       **config['data'])

    model2 = Model(net_test_loss_eqn,
                   loss_fn=None,
                   eval_network=net_test_loss_eqn,
                   metrics={'test mean eqn loss': nn.Loss()})

    model1.build(valid_dataset=test_dataset1)
    model2.build(valid_dataset=test_dataset2)
    with profiler(comment='Eval PINO3d'):
        output1 = model1.eval(test_dataset1)
        output2 = model2.eval(test_dataset2)
    print(output1)
    print(output2)


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

    if model_name == 'pino2d':
        test_darcy(config=eval_config, profiler=MyProfiler)
    elif model_name == 'pino3d':
        test_3d(config=eval_config, profiler=MyProfiler)
    else:
        raise ValueError(
            f'FNO model name config {model_name} is not supported')
