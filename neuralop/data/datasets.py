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

import math

import scipy.io
import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
import mindspore.dataset as ds
from mindspore import Tensor


class Burgers:

    def __init__(self, train_path, test_path, *args, **kwargs):
        train_dataset = scipy.io.loadmat(train_path)
        test_dataset = scipy.io.loadmat(test_path)
        train_data = train_dataset['a']
        train_label = train_dataset['u']
        train_data = train_data[..., np.newaxis]
        self.train_data = train_data.astype(np.float32)
        self.train_label = train_label.astype(np.float32)
        test_data = test_dataset['a']
        test_label = test_dataset['u']
        test_data = test_data[..., np.newaxis]
        self.test_data = test_data.astype(np.float32)
        self.test_label = test_label.astype(np.float32)

    def gen_train(self, ntrain, sub, batch_size, *args, **kwargs):
        data = self.train_data[:ntrain, ::sub]
        label = self.train_label[:ntrain, ::sub]
        return ds.NumpySlicesDataset(data={
            'data': data,
            'label': label
        }).batch(batch_size, drop_remainder=True)

    def gen_test(self, ntest, sub, batch_size, *args, **kwargs):
        data = self.test_data[-ntest:, ::sub]
        label = self.test_label[-ntest:, ::sub]
        return ds.NumpySlicesDataset(data={
            'data': data,
            'label': label
        }).batch(batch_size, drop_remainder=True)


class DarcyFlowFNO:

    def __init__(self, train_path, test_path, *args, **kwargs):
        train_dataset = scipy.io.loadmat(train_path)
        test_dataset = scipy.io.loadmat(test_path)
        self.train_data = train_dataset['coeff'].astype(np.float32)
        self.train_label = train_dataset['sol'].astype(np.float32)
        self.test_data = test_dataset['coeff'].astype(np.float32)
        self.test_label = test_dataset['sol'].astype(np.float32)
        self.normalizer = None

    def gen_train(self, ntrain, r, h, batch_size, *args, **kwargs):
        data = self.train_data[:ntrain, ::r, ::r][:, :h, :h]
        label = self.train_label[:ntrain, ::r, ::r][:, :h, :h]
        self.normalizer = UnitGaussianNormalizer(data)
        data = self.normalizer.encode(data)
        data = data[..., np.newaxis]
        return ds.NumpySlicesDataset(data={
            'data': data,
            'label': label
        }).batch(batch_size, drop_remainder=True)

    def gen_test(self, ntest, r, h, batch_size, *args, **kwargs):
        data = self.test_data[:ntest, ::r, ::r][:, :h, :h]
        label = self.test_label[:ntest, ::r, ::r][:, :h, :h]
        if isinstance(self.normalizer, UnitGaussianNormalizer):
            data = self.normalizer.encode(data)
        else:
            raise AttributeError(
                f'call train_dataset() before test_dataset to initialze normalizer'
            )
        data = data[..., np.newaxis]
        return ds.NumpySlicesDataset(data={
            'data': data,
            'label': label
        }).batch(batch_size, drop_remainder=True)


class NS2dTime:

    def __init__(self, train_path, test_path, *args, **kwargs):
        train_dataset = scipy.io.loadmat(train_path)
        self.train_dataset = train_dataset['u'].astype(np.float32)
        test_dataset = scipy.io.loadmat(train_path)
        self.test_dataset = test_dataset['u'].astype(np.float32)

    def gen_train(self, ntrain, sub, t, t_in, batch_size, *args, **kwargs):
        data = self.train_dataset[:ntrain, ::sub, ::sub, :t_in]
        label = self.train_dataset[:ntrain, ::sub, ::sub, t_in:t + t_in]
        return ds.NumpySlicesDataset(data={
            'data': data,
            'label': label
        }).batch(batch_size, drop_remainder=True)

    def gen_test(self, ntest, sub, t, t_in, batch_size, *args, **kwargs):
        data = self.test_dataset[-ntest:, ::sub, ::sub, :t_in]
        label = self.test_dataset[-ntest:, ::sub, ::sub, t_in:t + t_in]
        return ds.NumpySlicesDataset(data={
            'data': data,
            'label': label
        }).batch(batch_size, drop_remainder=True)


class NSFNO:

    def __init__(self, train_path, test_path, *args, **kwargs):
        train_dataset = scipy.io.loadmat(train_path)
        self.train_dataset = train_dataset['u'].astype(np.float32)
        test_dataset = scipy.io.loadmat(test_path)
        self.test_dataset = test_dataset['u'].astype(np.float32)
        self.normalizer = None

    def gen_train(self, ntrain, nx, sub, t, t_in, batch_size, *args, **kwargs):
        data = self.train_dataset[:ntrain, ::sub, ::sub, :t_in]
        label = self.train_dataset[:ntrain, ::sub, ::sub, t_in:t + t_in]
        self.normalizer = UnitGaussianNormalizer(data)
        data = self.normalizer.encode(data)
        data = data.reshape(ntrain, nx, nx, 1, t_in)
        data = np.tile(data, (1, 1, 1, t, 1))
        return ds.NumpySlicesDataset(data={
            'data': data,
            'label': label
        }).batch(batch_size, drop_remainder=True)

    def gen_test(self, ntest, nx, sub, t, t_in, batch_size, *args, **kwargs):
        data = self.test_dataset[-ntest:, ::sub, ::sub, :t_in]
        label = self.test_dataset[-ntest:, ::sub, ::sub, t_in:t + t_in]
        if isinstance(self.normalizer, UnitGaussianNormalizer):
            data = self.normalizer.encode(data)
            data = data.reshape(ntest, nx, nx, 1, t_in)
            data = np.tile(data, (1, 1, 1, t, 1))
        else:
            raise AttributeError(
                'call train_dataset() before test_dataset to initialze normalizer'
            )
        return ds.NumpySlicesDataset(data={
            'data': data,
            'label': label
        }).batch(batch_size, drop_remainder=True)


class DarcyFlowPINO:

    def __init__(self, train_path, test_path, *args, **kwargs):
        train_dataset = scipy.io.loadmat(train_path)
        test_dataset = scipy.io.loadmat(test_path)
        self.train_data = train_dataset['coeff'].astype(np.float32)[...,
                                                                    np.newaxis]
        self.train_label = train_dataset['sol'].astype(np.float32)
        self.test_data = test_dataset['coeff'].astype(np.float32)[...,
                                                                  np.newaxis]
        self.test_label = test_dataset['sol'].astype(np.float32)
        self.mesh = None

    @staticmethod
    def mollifier(nx, sub, *args, **kwargs):
        s = int(nx // sub) + 1
        mesh = ms2dgrid(s, s)
        return ms.ops.sin(np.pi * mesh[..., 0]) * ms.ops.sin(
            np.pi * mesh[..., 1]) * 0.001

    def gen_grid(self, nx, sub, *args, **kwargs):
        s = int(nx // sub) + 1
        self.mesh = np2dgrid(s, s)

    def gen_train(self, nx, sub, offset, ntrain, batch_size, *args, **kwargs):
        data = self.train_data[offset:offset + ntrain, ::sub, ::sub, :]
        if self.mesh is None:
            self.gen_grid(nx, sub)
        data = np.concatenate((data, np.tile(self.mesh, (ntrain, 1, 1, 1))),
                              axis=3)
        label = self.train_label[offset:offset + ntrain, ::sub, ::sub]
        return ds.NumpySlicesDataset(data={
            'data': data,
            'label': label
        }).batch(batch_size, drop_remainder=True)

    def gen_test(self, nx, sub, offset, ntest, batch_size, *args, **kwargs):
        data = self.test_data[offset:offset + ntest, ::sub, ::sub, :]
        if self.mesh is None:
            self.gen_grid(nx, sub)
        data = np.concatenate((data, np.tile(self.mesh, (ntest, 1, 1, 1))),
                              axis=3)
        label = self.test_label[offset:offset + ntest, ::sub, ::sub]
        return ds.NumpySlicesDataset(data={
            'data': data,
            'label': label
        }).batch(batch_size, drop_remainder=True)


class NSPINO:

    def __init__(self, datapath, nx, nt, sub, sub_t, time_interval, *args,
                 **kwargs):
        self.nx = nx
        self.nt = nt
        self.sub = sub
        self.sub_t = sub_t
        self.s = nx // sub
        self.t = int(nt * time_interval) // sub_t + 1
        self.time_scale = time_interval
        self.dataset = np.load(datapath)
        self.eqn_data = None
        self.data = None
        self.label = None
    
    @staticmethod
    def extract(data):
        t = data.shape[1] // 2
        interval = data.shape[1] // 4
        n = data.shape[0]
        new_data = np.zeros((4 * n - 1, t + 1, data.shape[2], data.shape[3]))
        for i in range(n):
            for j in range(4):
                if i == n - 1 and j == 3:
                    break
                if j != 3:
                    new_data[i * 4 + j] = data[i, interval * j:interval * j +
                                               t + 1]
                else:
                    temp = data[i, interval * j:interval * j + interval]
                    new_data[i * 4 + j, 0:interval] = temp
                    new_data[i * 4 + j, interval:t + 1] = data[i + 1,
                                                               0:interval + 1]
        return new_data

    def gen_dataset(self, batch_size, shuffle):
        return ds.NumpySlicesDataset(data={
            'data': self.data,
            'label': self.label
        }, shuffle=shuffle).batch(batch_size, drop_remainder=True)

    def gen_eqnset(self, batch_size):
        return ds.GeneratorDataset(self.eqn_data, column_names=[
                                   'data']).batch(batch_size)

    def gen_gaussianrf(self, batch_size, nx_eqn, nt_eqn, time_interval, *args,
                       **kwargs):
        gr_sampler = GaussianRF(2, nx_eqn, 2 * math.pi, alpha=2.5, tau=7.)
        self.eqn_data = online_loader(gr_sampler,
                                      nx_eqn,
                                      nt_eqn,
                                      time_interval,
                                      batch_size)

    def append_mesh(self, nsample, train=True, offset=0):
        if train:
            self.dataset = self.dataset[:nsample // 4 + \
                1, ::self.sub_t, ::self.sub, ::self.sub]
            self.dataset = self.extract(self.dataset)
            self.dataset = np.transpose(self.dataset, (0, 2, 3, 1))
            data = self.dataset[offset:offset + nsample, :, :,
                                0].reshape(nsample, self.s, self.s)
            label = self.dataset[offset:offset + nsample].reshape(
                nsample, self.s, self.s, self.t)
        else:
            self.dataset = self.dataset[-nsample // 4 - \
                1:, ::self.sub_t, ::self.sub, ::self.sub]
            self.dataset = self.extract(self.dataset)
            self.dataset = np.transpose(self.dataset, (0, 2, 3, 1))
            data = self.dataset[-nsample:, :, :,
                                0].reshape(nsample, self.s, self.s)
            label = self.dataset[-nsample:].reshape(nsample, self.s, self.s,
                                                    self.t)

        data = data.reshape(nsample, self.s, self.s, 1, 1)
        data = np.tile(data, (1, 1, 1, self.t, 1))
        gridx, gridy, gridt = get_grid3d(self.s,
                                         self.t,
                                         time_scale=self.time_scale)
        gridx = np.tile(gridx, (nsample, 1, 1, 1, 1))
        gridy = np.tile(gridy, (nsample, 1, 1, 1, 1))
        gridt = np.tile(gridt, (nsample, 1, 1, 1, 1))
        data = np.concatenate((gridx, gridy, gridt, data), axis=-1)

        self.data = data.astype(np.float32)
        self.label = label.astype(np.float32)


class GaussianRF:

    def __init__(self,
                 dim,
                 size,
                 length=1.0,
                 alpha=2.0,
                 tau=3.0,
                 sigma=None,
                 boundary="periodic",
                 constant_eig=False):

        self.dim = dim

        if sigma is None:
            sigma = tau**(0.5 * (2 * alpha - self.dim))

        k_max = size // 2

        const = (4 * (math.pi**2)) / (length**2)

        if dim == 1:
            k = np.concatenate(
                (np.arange(start=0, stop=k_max,
                           step=1), np.arange(start=-k_max, stop=0, step=1)),
                axis=0)

            self.sqrt_eig = size * math.sqrt(2.0) * sigma * (
                (const * (k**2) + tau**2)**(-alpha / 2.0))

            if constant_eig:
                self.sqrt_eig[0] = size * sigma * (tau**(-alpha))
            else:
                self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = np.concatenate(
                (np.arange(start=0, stop=k_max,
                           step=1), np.arange(start=-k_max, stop=0, step=1)),
                axis=0)
            wavenumers = np.tile(wavenumers, (size, 1))

            k_x = wavenumers.transpose((1, 0))
            k_y = wavenumers

            self.sqrt_eig = (size**2) * math.sqrt(2.0) * sigma * (
                (const * (k_x**2 + k_y**2) + tau**2)**(-alpha / 2.0))

            if constant_eig:
                self.sqrt_eig[0, 0] = (size**2) * sigma * (tau**(-alpha))
            else:
                self.sqrt_eig[0, 0] = 0.0

        elif dim == 3:
            wavenumers = np.concatenate(
                (np.arange(start=0, stop=k_max,
                           step=1), np.arange(start=-k_max, stop=0, step=1)),
                axis=0)
            wavenumers = np.tile(wavenumers, (size, size, 1))

            k_x = wavenumers.transpose((0, 2, 1))
            k_y = wavenumers
            k_z = wavenumers.transpose((2, 1, 0))

            self.sqrt_eig = (size**3) * math.sqrt(2.0) * sigma * (
                (const * (k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha / 2.0))

            if constant_eig:
                self.sqrt_eig[0, 0, 0] = (size**3) * sigma * (tau**(-alpha))
            else:
                self.sqrt_eig[0, 0, 0] = 0.0

        self.size = []
        for _ in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, n):

        coeff_re = np.random.randn(n, *self.size)
        coeff_im = np.random.randn(n, *self.size)
        coeff = (coeff_re + 1j * coeff_im).astype(np.complex64)
        coeff = self.sqrt_eig * coeff

        u = np.fft.irfftn(coeff, self.size)
        return u


class UnitGaussianNormalizer(object):

    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0, ddof=1)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        std = self.std + self.eps
        mean = self.mean
        x = (x * std) + mean
        return x


def online_loader(sampler, s, t, time_scale, batch_size):
    while True:
        u_0 = sampler.sample(batch_size)
        a = convert_ic(u_0, batch_size, s, t, time_scale=time_scale)
        a = np.squeeze(a, 0)
        yield a.astype(np.float32)


def get_forcing(s):
    x_1 = Tensor(np.linspace(0, 2 * np.pi, s + 1)[:-1],
                dtype=ms.float32).reshape(s, 1)
    x_1 = mnp.tile(x_1, (1, s))
    x_2 = Tensor(np.linspace(0, 2 * np.pi, s + 1)[:-1],
                dtype=ms.float32).reshape(1, s)
    x_2 = mnp.tile(x_2, (s, 1))
    return -4 * (ms.ops.cos(4 * x_2)).reshape(1, s, s, 1)


def ms2dgrid(num_x, num_y, bot=(0, 0), top=(1, 1)):
    x_bot, y_bot = bot
    x_top, y_top = top
    x_arr = np.linspace(start=x_bot, stop=x_top, num=num_x)
    y_arr = np.linspace(start=y_bot, stop=y_top, num=num_y)
    xx, yy = np.meshgrid(x_arr, y_arr)
    mesh = np.stack([xx.T, yy.T], axis=2)
    return Tensor(mesh, ms.float32)


def np2dgrid(num_x, num_y, bot=(0, 0), top=(1, 1)):
    x_bot, y_bot = bot
    x_top, y_top = top
    x_arr = np.linspace(start=x_bot, stop=x_top, num=num_x)
    y_arr = np.linspace(start=y_bot, stop=y_top, num=num_y)
    xx, yy = np.meshgrid(x_arr, y_arr)
    mesh = np.stack([xx.T, yy.T], axis=2)
    return mesh.astype(np.float32)


def get_grid3d(s, t, time_scale=1.0):
    gridx = np.linspace(0, 1, s + 1)[:-1].astype(np.float32)
    gridx = gridx.reshape(1, s, 1, 1, 1)
    gridx = np.tile(gridx, (1, 1, s, t, 1))

    gridy = np.linspace(0, 1, s + 1)[:-1].astype(np.float32)
    gridy = gridy.reshape(1, 1, s, 1, 1)
    gridy = np.tile(gridy, (1, s, 1, t, 1))

    gridt = np.linspace(0, 1 * time_scale, t).astype(np.float32)
    gridt = gridt.reshape(1, 1, 1, t, 1)
    gridt = np.tile(gridt, (1, s, s, 1, 1))
    return gridx, gridy, gridt


def convert_ic(u0, n, s, t, time_scale=1.0):
    u0 = u0.reshape(n, s, s, 1, 1)
    u0 = np.tile(u0, (1, 1, 1, t, 1))
    gridx, gridy, gridt = get_grid3d(s, t, time_scale=time_scale)
    gridx = np.tile(gridx, (n, 1, 1, 1, 1))
    gridy = np.tile(gridy, (n, 1, 1, 1, 1))
    gridt = np.tile(gridt, (n, 1, 1, 1, 1))
    a_data = np.concatenate((gridx, gridy, gridt, u0), axis=-1)
    return a_data


def infinite_loader(loader):
    while True:
        for item in loader:
            yield item
