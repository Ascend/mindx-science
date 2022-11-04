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
from mindspore import DynamicLossScaleManager, Model, context, nn
from mindspore.train.callback import LearningRateScheduler, LossMonitor

from architecture.fourier2d import FNN2d
from architecture.fourier3d import FNN3d
from data.datasets import NSPINO, DarcyFlowPINO, get_forcing, infinite_loader
from loss.losses import (DarcyLoss, DarcyTrainLoss, LPLoss, NSTrainDataLoss,
                         NSTrainEqnLoss, PINOLoss3D)
from utils.utils import MyCallback, MyProfiler, load_model, step_lr_wrapper

ms.set_seed(1234)

context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')

mix_prec_config1 = {
    'amp_level': 'O3',
    'loss_scale_manager': DynamicLossScaleManager(),
    'keep_batchnorm_fp32': True
}

mix_prec_config2 = {
    'level': 'O3',
    'loss_scale_manager': DynamicLossScaleManager(),
    'keep_batchnorm_fp32': True
}


def train_darcy(config, profiler, *args, **kwargs):
    epoch = config['train']['epoch']
    base_lr = config['train']['base_lr']
    mix_prec = config['train']['mix_prec']
    data_base = DarcyFlowPINO(**config['data'])
    train_dataset = data_base.gen_train(**config['data'])
    test_dataset = data_base.gen_test(**config['data'])
    mollifier = data_base.mollifier(**config['data'])

    with profiler(**config['train']):
        net = FNN2d(**config['model'])
        if 'load_path' in config['train']:
            load_path = config['train']['load_path']
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

        net_with_trainloss = DarcyTrainLoss(net=net,
                                            mollifier=mollifier,
                                            lploss=LPLoss(),
                                            darcyloss=DarcyLoss(),
                                            **config['data'])
        if mix_prec:
            model = Model(net_with_trainloss,
                          loss_fn=None,
                          optimizer=optimizer,
                          **mix_prec_config1)
        else:
            model = Model(net_with_trainloss,
                          loss_fn=None,
                          optimizer=optimizer)
        model.train(epoch=epoch,
                    train_dataset=train_dataset,
                    callbacks=callbacks)
    print('Train PINO-Darcy Done!')


# train ns
def train_3d(config, profiler, *args, **kwargs):
    epoch = config['train']['epoch']
    base_lr = config['train']['base_lr']
    weight_decay = config['train']['weight_decay']
    batch_size = config['data']['batch_size']
    mix_prec = config['train']['mix_prec']
    train_path = config['data']['train_path']
    ntrain = config['data']['ntrain']
    re = config['data']['Re']
    data_s, eqn_s = config['data']['nx_data'], config['data']['nx_eqn']
    time_interval = config['data']['time_interval']
    num_data_iter = config['train']['num_data_iter']
    num_eqn_iter = config['train']['num_eqn_iter']
    shuffle = config['data']['shuffle']
    if 'save_per_epoch' in config['train']:
        save_per_epoch = config['train']['save_per_epoch']
    else:
        save_per_epoch = epoch
    if 'save_path' in config['train']:
        save_path = config['train']['save_path']
    else:
        save_path = None

    train_base = NSPINO(datapath=train_path, **config['data'])
    train_base.append_mesh(nsample=ntrain)
    train_dataset = train_base.gen_dataset(
        batch_size=batch_size, shuffle=shuffle)
    train_loader = infinite_loader(train_dataset.create_dict_iterator())
    train_base.gen_gaussianrf(**config['data'])
    eqn_dataset = train_base.gen_eqnset(batch_size=batch_size)
    eqn_loader = infinite_loader(eqn_dataset.create_dict_iterator())

    forcing_1 = get_forcing(data_s)
    forcing_2 = get_forcing(eqn_s)
    v = float(1 / re)

    with profiler(**config['train'], comment='Train PINO3d'):
        net = FNN3d(**config['model'])

        optimizer = nn.Adam(params=net.trainable_params(),
                            learning_rate=base_lr,
                            weight_decay=weight_decay)
        net_with_loss_data = NSTrainDataLoss(net=net,
                                             lploss=LPLoss(),
                                             nsloss=PINOLoss3D(
                                                 forcing_1, v, time_interval),
                                             **config['data'],
                                             **config['train'])

        net_with_loss_eqn = NSTrainEqnLoss(net=net,
                                           nsloss=PINOLoss3D(
                                               forcing_2, v, time_interval),
                                           **config['data'],
                                           **config['train'])

        if mix_prec:
            model_data = ms.build_train_network(
                net_with_loss_data, optimizer, **mix_prec_config2)
            model_eqn = ms.build_train_network(
                net_with_loss_eqn, optimizer, **mix_prec_config2)
        else:
            model_data = ms.build_train_network(net_with_loss_data, optimizer)
            model_eqn = ms.build_train_network(net_with_loss_eqn, optimizer)

        for niter in range(1, epoch + 1):
            print(f'===Epoch:[{niter}/{epoch}]===', flush=True)
            train_loss = 0.0
            err_eqn = 0.0
            for _ in range(num_data_iter):
                data = next(train_loader)
                total_loss = model_data(data["data"], data["label"])
                if isinstance(total_loss, tuple):
                    train_loss += total_loss[0]
                else:
                    total_loss += total_loss

            for _ in range(num_eqn_iter):
                data = next(eqn_loader)
                eqn_loss = model_eqn(data["data"])
                if isinstance(eqn_loss, tuple):
                    err_eqn += eqn_loss[0]
                else:
                    err_eqn += eqn_loss

            train_loss /= (num_data_iter)
            err_eqn /= (num_eqn_iter)
            print(
                f"Data loss: {train_loss} Equation loss: {err_eqn}",
                flush=True)
            if save_path and niter % save_per_epoch == 0:
                ms.save_checkpoint(net, save_path)
    print('Train PINO-NS Done!')


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

    if model_name == 'pino2d':
        train_darcy(config=train_config, profiler=MyProfiler)
    elif model_name == 'pino3d':
        train_3d(config=train_config, profiler=MyProfiler)
    else:
        raise ValueError(
            f'PINO model name config {model_name} is not supported')
