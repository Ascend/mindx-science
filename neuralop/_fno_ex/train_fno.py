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
from mindspore import nn, context, Model, DynamicLossScaleManager
from mindspore.train.callback import LossMonitor, LearningRateScheduler

from data.datasets import NS2dTime, Burgers, DarcyFlowFNO, NSFNO
from architecture.fourier1d import FNO1d
from architecture.fourier3d import FNO3d
from architecture.fourier2d import FNO2d, FNO2dtime
from loss.losses import LPLoss, FNO2dTimeTrainLoss, FNO2dTimeL2Loss
from utils.utils import step_lr_wrapper, MyCallback, load_model, MyProfiler

ms.set_seed(1234)

context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')

mix_prec_config = {
    'amp_level': 'O3',
    'loss_scale_manager': DynamicLossScaleManager(),
    'keep_batchnorm_fp32': True
}


def train_1d(config, profiler, *args, **kwargs):
    epoch = config['train']['epoch']
    base_lr = config['train']['base_lr']
    mix_prec = config['train']['mix_prec']

    data_base = Burgers(**config['data'])
    train_dataset = data_base.gen_train(**config['data'])
    test_dataset = data_base.gen_test(**config['data'])

    with profiler(comment='Train FNO1d', **config['train']):
        net = FNO1d(**config['model'])
        if 'load_path' in config['train']:
            load_path = config['train']['load_path']
            load_model(net, load_path)
            print(f'Weights loaded from {load_path}')

        loss = LPLoss()
        optimizer = nn.Adam(params=net.trainable_params(),
                            learning_rate=base_lr,
                            weight_decay=config['train']['weight_decay'])
        callbacks = [
            LossMonitor(),
            MyCallback(**config['train']),
            LearningRateScheduler(
                step_lr_wrapper(**config['data'], **config['train']))
        ]
        if mix_prec:
            model = Model(net,
                          loss_fn=loss,
                          optimizer=optimizer,
                          metrics={'test mean l2 loss': nn.Loss()},
                          **mix_prec_config)
        else:
            model = Model(net,
                          loss_fn=loss,
                          optimizer=optimizer,
                          metrics={'test mean L2 loss': nn.Loss()})
        model.train(epoch=epoch,
                    train_dataset=train_dataset,
                    callbacks=callbacks)
    print('Train FNO1d Done!')
    if not config['train']['profile']:
        with profiler(comment='Eval FNO1d'):
            print(model.eval(test_dataset))


def train_2d(config, profiler, *args, **kwargs):
    epoch = config['train']['epoch']
    base_lr = config['train']['base_lr']
    mix_prec = config['train']['mix_prec']

    data_base = DarcyFlowFNO(**config['data'])
    train_dataset = data_base.gen_train(**config['data'])
    test_dataset = data_base.gen_test(**config['data'])

    with profiler(comment='Train FNO2d', **config['train']):
        net = FNO2d(**config['model'])
        if 'load_path' in config['train']:
            load_path = config['train']['load_path']
            load_model(net, load_path)
            print(f'Weights loaded from {load_path}')

        loss = LPLoss()
        optimizer = nn.Adam(params=net.trainable_params(),
                            learning_rate=base_lr,
                            weight_decay=config['train']['weight_decay'])
        callbacks = [
            LossMonitor(),
            MyCallback(**config['train']),
            LearningRateScheduler(
                step_lr_wrapper(**config['data'], **config['train']))
        ]
        if mix_prec:
            model = Model(net,
                          loss_fn=loss,
                          optimizer=optimizer,
                          metrics={'test mean L2 loss': nn.Loss()},
                          **mix_prec_config)
        else:
            model = Model(net,
                          loss_fn=loss,
                          optimizer=optimizer,
                          metrics={'test mean L2 loss': nn.Loss()})
        model.train(epoch=epoch,
                    train_dataset=train_dataset,
                    callbacks=callbacks)
    print('Train FNO2d Done!')
    if not config['train']['profile']:
        with profiler(comment='Eval FNO2d'):
            print(model.eval(test_dataset))


def train_2dtime(config, profiler, *args, **kwargs):
    epoch = config['train']['epoch']
    base_lr = config['train']['base_lr']
    mix_prec = config['train']['mix_prec']
    data_base = NS2dTime(**config['data'])
    train_dataset = data_base.gen_train(**config['data'])
    test_dataset = data_base.gen_test(**config['data'])

    with profiler(comment='Train FNO2dtime', **config['train']):
        net = FNO2dtime(**config['data'], **config['model'])
        if 'load_path' in config['train']:
            load_path = config['train']['ckpt']
            load_model(net, load_path)
            print(f'Weights loaded from {load_path}')

        optimizer = nn.Adam(params=net.trainable_params(),
                            learning_rate=base_lr,
                            weight_decay=config['train']['weight_decay'])
        callbacks = [
            LossMonitor(),
            MyCallback(**config['train']),
            LearningRateScheduler(
                step_lr_wrapper(**config['data'], **config['train']))
        ]
        net_with_trainloss = FNO2dTimeTrainLoss(net=net,
                                                lploss=LPLoss(),
                                                **config['data'])
        if mix_prec:
            model = Model(net_with_trainloss,
                          loss_fn=None,
                          optimizer=optimizer,
                          **mix_prec_config)
        else:
            model = Model(
                net_with_trainloss,
                loss_fn=None,
                optimizer=optimizer)
        model.train(epoch=epoch,
                    train_dataset=train_dataset,
                    callbacks=callbacks)
    print('Train FNO2dtime Done!')
    net_with_l2loss = FNO2dTimeL2Loss(net=net,
                                      lploss=LPLoss(),
                                      **config['data'])
    model = Model(net_with_l2loss,
                  loss_fn=None,
                  eval_network=net_with_l2loss,
                  metrics={'test mean L2 loss': nn.Loss()})
    if not config['train']['profile']:
        with profiler(comment='Eval FNO2dtime'):
            print(model.eval(test_dataset))


def train_3d(config, profiler, *args, **kwargs):
    epoch = config['train']['epoch']
    base_lr = config['train']['base_lr']
    mix_prec = config['train']['mix_prec']

    data_base = NSFNO(**config['data'])
    train_dataset = data_base.gen_train(**config['data'])
    test_dataset = data_base.gen_test(**config['data'])

    with profiler(comment='Train FNO3d', **config['train']):
        net = FNO3d(**config['data'], **config['model'])
        if 'load_path' in config['train']:
            load_path = config['train']['ckpt']
            load_model(net, load_path)
            print(f'Weights loaded from {load_path}')

        loss = LPLoss()
        optimizer = nn.Adam(params=net.trainable_params(),
                            learning_rate=base_lr,
                            weight_decay=config['train']['weight_decay'])
        callbacks = [
            LossMonitor(),
            MyCallback(**config['train']),
            LearningRateScheduler(
                step_lr_wrapper(**config['data'], **config['train']))
        ]
        if mix_prec:
            model = Model(net,
                          loss_fn=loss,
                          optimizer=optimizer,
                          metrics={'test mean L2 loss': nn.Loss()},
                          **mix_prec_config)
        else:
            model = Model(net,
                          loss_fn=loss,
                          optimizer=optimizer,
                          metrics={'test mean L2 loss': nn.Loss()})
        model.train(epoch=epoch,
                    train_dataset=train_dataset,
                    callbacks=callbacks)
    print('Train FNO3d Done!')
    if not config['train']['profile']:
        with profiler(comment='Eval FNO3d'):
            print(model.eval(test_dataset))


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path',
                        type=str,
                        help='Path to the configuration file')
    options = parser.parse_args()

    config_path = options.config_path
    with open(config_path, 'r', encoding='UTF-8') as stream:
        train_config = yaml.safe_load(stream)

    print(f'train configurations:\n{train_config}')

    assert ('name' in train_config['model'])
    model_name = train_config['model']['name']

    if model_name == 'fno1d':
        train_1d(config=train_config, profiler=MyProfiler)
    elif model_name == 'fno2d':
        train_2d(config=train_config, profiler=MyProfiler)
    elif model_name == 'fno2dtime':
        train_2dtime(config=train_config, profiler=MyProfiler)
    elif model_name == 'fno3d':
        train_3d(config=train_config, profiler=MyProfiler)
    else:
        raise ValueError(
            f'FNO model name config {model_name} is not supported')
