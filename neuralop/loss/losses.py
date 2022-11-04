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

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import nn
from mindspore.ops import functional as F

from architecture.fft_ops_1 import FFT2, IRFFT2


class LPLoss(nn.LossBase):
    def __init__(self, reduction='mean'):
        super(LPLoss, self).__init__()
        self.reduction = reduction
        self.norm_op = ms.ops.LpNorm(axis=-1)
        self.mean_op = ms.ops.ReduceMean()
        self.sum_op = ms.ops.ReduceSum()
        self.reshape_op = ms.ops.Reshape()

    def rel(self, x, y):
        x = self.reshape_op(x, (x.shape[0], -1))
        y = self.reshape_op(y, (y.shape[0], -1))
        diff_norms = self.norm_op(x - y)
        y_norms = self.norm_op(y)
        ret = diff_norms / y_norms
        if self.reduction == 'mean':
            ret = self.mean_op(ret, 0)
        elif self.reduction == 'sum':
            ret = self.sum_op(ret, 0)
        return ret

    def construct(self, predict, target):
        return self.rel(predict, target)


class FNO2dTimeTrainLoss(nn.Cell):
    def __init__(self, net, lploss, t, step, *args, **kwargs):
        super(FNO2dTimeTrainLoss, self).__init__()
        self.net = net
        self.lploss = lploss
        self.t = t
        self.step = step
        self.reshape = ms.ops.Reshape()

    def construct(self, data, label):
        loss = 0.
        xx = data
        yy = label
        t = self.t
        step = self.step

        for nt in range(0, t, step):
            y = yy[..., nt:nt + step]
            im = self.net(xx)
            loss += self.lploss(im.astype(y.dtype), y)
            xx = ms.ops.Concat(-1)((xx[..., step:], im.astype(xx.dtype)))
        return loss


class FNO2dTimeL2Loss(nn.Cell):
    def __init__(self, net, lploss, t, step, *args, **kwargs):
        super(FNO2dTimeL2Loss, self).__init__()
        self.net = net
        self.loss = lploss
        self.t = t
        self.step = step

    def construct(self, data, label):
        loss = 0.
        xx = data
        t = self.t
        step = self.step

        pred = self.net(xx)
        xx = ms.ops.Concat(-1)((xx[..., step:], pred.astype(xx.dtype)))
        for _ in range(step, t, step):
            im = self.net(xx)
            pred = ms.ops.Concat(-1)((pred, im))
            xx = ms.ops.Concat(-1)((xx[..., step:], im.astype(xx.dtype)))
        loss = self.loss(pred.astype(label.dtype), label)
        return loss


class DarcyLoss(nn.LossBase):
    def __init__(self):
        super(DarcyLoss, self).__init__()
        self.reshape = ms.ops.Reshape()
        self.ones = ms.ops.Ones()
        self.lploss = LPLoss()

    def fdm_darcy(self, u, a, size, d=1):
        dx_recip = (size - 1) / d
        dy_recip = dx_recip
        ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) * (0.5 * dx_recip)
        uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) * (0.5 * dy_recip)
        a = a[:, 1:-1, 1:-1]
        aux = a * ux
        auy = a * uy
        auxx = (aux[:, 2:, 1:-1] - aux[:, :-2, 1:-1]) * (0.5 * dx_recip)
        auyy = (auy[:, 1:-1, 2:] - auy[:, 1:-1, :-2]) * (0.5 * dy_recip)
        du = - (auxx + auyy)
        return du

    def construct(self, u, a):
        batchsize = u.shape[0]
        size = u.shape[1]
        u = self.reshape(u, (batchsize, size, size))
        a = self.reshape(a, (batchsize, size, size))
        du = self.fdm_darcy(u, a, size)
        f = self.ones(du.shape, ms.float32)
        loss_f = self.lploss(du, f)
        return loss_f


class DarcyTrainLoss(nn.Cell):
    def __init__(
            self,
            net,
            mollifier,
            lploss,
            darcyloss,
            data_weight,
            f_weight,
            *args,
            **kwargs):
        super(DarcyTrainLoss, self).__init__()
        self.net = net
        self.mollifier = mollifier
        self.data_weight = data_weight
        self.f_weight = f_weight
        self.lploss = lploss
        self.darcyloss = darcyloss
        self.reshape = ms.ops.Reshape()

    def construct(self, data, label):
        a = data[..., 0]
        predict = self.net(data)
        predict = self.reshape(predict, label.shape)
        predict = predict * self.mollifier
        f_loss = self.f_weight * self.darcyloss(predict, a)
        data_loss = self.data_weight * self.lploss(predict, label)
        return f_loss + data_loss


class DarcyDataLoss(nn.Cell):
    def __init__(self, net, mollifier, lploss, *args, **kwargs):
        super(DarcyDataLoss, self).__init__()
        self.net = net
        self.mollifier = mollifier
        self.reshape = ms.ops.Reshape()
        self.lploss = lploss

    def construct(self, data, label):
        predict = self.net(data)
        predict = self.reshape(predict, label.shape)
        predict = predict * self.mollifier
        data_loss = self.lploss(predict, label)
        return data_loss


class DarcyEqnLoss(nn.Cell):
    def __init__(self, net, mollifier, darcyloss, *args, **kwargs):
        super(DarcyEqnLoss, self).__init__()
        self.net = net
        self.mollifier = mollifier
        self.darcyloss = darcyloss
        self.reshape = ms.ops.Reshape()

    def construct(self, data, label):
        a = data[..., 0]
        predict = self.net(data)
        predict = self.reshape(predict, label.shape)
        predict = predict * self.mollifier
        f_loss = self.darcyloss(predict, a)
        return f_loss


class PINOLoss3D(nn.LossBase):
    def __init__(self, forcing, v, t_interval):
        super(PINOLoss3D, self).__init__()
        self.reshape = ms.ops.Reshape()
        self.ones = ms.ops.Ones()
        self.lploss = LPLoss()
        self.forcing = forcing
        self.v = v
        self.t_interval = t_interval

    def ifft2_me(self, re, im):
        re = ms.ops.transpose(F.cast(re, ms.float32), (0, 3, 1, 2))
        im = ms.ops.transpose(F.cast(im, ms.float32), (0, 3, 1, 2))
        s = (re.shape[2], 2 * (re.shape[3] - 1))
        irfft2 = IRFFT2(s)
        ifft2_out = irfft2(F.cast(re, ms.float32), F.cast(im, ms.float32))
        ifft2_out = ms.ops.transpose(
            F.cast(ifft2_out, ms.float32), (0, 2, 3, 1))
        return ifft2_out

    def fdm_ns_vorticity(self, w):
        batchsize = w.shape[0]
        nx = w.shape[1]
        ny = w.shape[2]
        nt = w.shape[3]
        w = self.reshape(w, (batchsize, nx, ny, nt))

        w = ms.ops.transpose(w, (0, 3, 1, 2))
        fft2 = FFT2()
        w = F.cast(w, ms.float32)
        w_im = mnp.zeros(w.shape)
        w_im = F.cast(w_im, ms.float32)
        w_h_re, w_h_im = fft2(w, w_im)
        w_h_re = ms.ops.transpose(F.cast(w_h_re, ms.float32), (0, 2, 3, 1))
        w_h_im = ms.ops.transpose(F.cast(w_h_im, ms.float32), (0, 2, 3, 1))
        w = ms.ops.transpose(F.cast(w, ms.float32), (0, 2, 3, 1))
        # Wavenumbers in y-direction
        k_max = nx // 2
        n = nx
        k_x = ms.ops.Concat(0)(
            (mnp.arange(
                start=0,
                stop=k_max,
                step=1),
                mnp.arange(
                start=-k_max,
                stop=0,
                step=1))).reshape(
            n,
            1)
        k_x = mnp.tile(k_x, (1, n)).reshape(1, n, n, 1)
        k_y = ms.ops.Concat(0)(
            (mnp.arange(
                start=0,
                stop=k_max,
                step=1),
                mnp.arange(
                start=-k_max,
                stop=0,
                step=1))).reshape(
            1,
            n)
        k_y = mnp.tile(k_y, (n, 1)).reshape(1, n, n, 1)

        # Negative Laplacian in Fourier space
        lap = (k_x ** 2 + k_y ** 2)
        lap = F.cast(lap, ms.float32)
        lap[0, 0, 0, 0] = F.cast(1.0, ms.float32)
        f_h_re = w_h_re / lap
        f_h_im = w_h_im / lap

        ux_h_re = -k_y * f_h_im
        ux_h_im = k_y * f_h_re

        uy_h_re = k_x * f_h_im
        uy_h_im = -k_x * f_h_re

        wx_h_re = -k_x * w_h_im
        wx_h_im = k_x * w_h_re

        wy_h_re = -k_y * w_h_im
        wy_h_im = k_y * w_h_re

        wlap_h_re = -lap * w_h_re
        wlap_h_im = -lap * w_h_im

        ux = self.ifft2_me(ux_h_re[:, :, :k_max + 1],
                           ux_h_im[:, :, :k_max + 1])
        # print(ux.shape) [ 1 64 64 65]
        uy = self.ifft2_me(uy_h_re[:, :, :k_max + 1],
                           uy_h_im[:, :, :k_max + 1])
        wx = self.ifft2_me(wx_h_re[:, :, :k_max + 1],
                           wx_h_im[:, :, :k_max + 1])
        wy = self.ifft2_me(wy_h_re[:, :, :k_max + 1],
                           wy_h_im[:, :, :k_max + 1])
        wlap = self.ifft2_me(
            wlap_h_re[:, :, :k_max + 1], wlap_h_im[:, :, :k_max + 1])
        dt = self.t_interval / (nt - 1)
        wt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)
        du = wt + (ux * wx + uy * wy - self.v * wlap)[..., 1:-1]
        return du

    def construct(self, u, u0):

        batchsize = u.shape[0]
        nx = u.shape[1]
        ny = u.shape[2]
        nt = u.shape[3]

        u = self.reshape(u, (batchsize, nx, ny, nt))

        u_in = u[:, :, :, 0]
        loss_ic = LPLoss(reduction='mean')(u_in, u0)

        du = self.fdm_ns_vorticity(u)
        f = mnp.tile(self.forcing, (batchsize, 1, 1, nt - 2))
        loss_f = self.lploss(F.cast(du, ms.float32), F.cast(f, ms.float32))

        return loss_ic, loss_f


class NSTrainDataLoss(nn.Cell):
    def __init__(
            self,
            net,
            lploss,
            nsloss,
            nx_data,
            nt_data,
            batch_size,
            xy_weight,
            f_weight,
            ic_weight,
            *args,
            **kwargs):
        super(NSTrainDataLoss, self).__init__()
        self.net = net
        self.batchsize = batch_size
        self.xy_weight = xy_weight
        self.f_weight = f_weight
        self.ic_weight = ic_weight
        self.reshape = ms.ops.Reshape()
        self.lploss = lploss
        self.nsloss = nsloss
        self.nx = nx_data
        self.nt = nt_data

    def construct(self, x, y):
        padding_shape = (x.shape[0], x.shape[1], x.shape[-3], 5, x.shape[-1])
        padding = mnp.zeros(padding_shape)
        x_in = ms.ops.Concat(-2)((x.astype(padding.dtype), padding))
        predict = self.net(x_in)
        predict = self.reshape(
            predict, (self.batchsize, self.nx, self.nx, self.nt + 5))
        predict = predict[..., :-
                          5].reshape(self.batchsize, self.nx, self.nx, self.nt)
        y = y.reshape(self.batchsize, self.nx, self.nx, self.nt)
        x = x[:, :, :, 0, -1]

        loss_l2 = self.lploss(predict, y)
        if self.ic_weight != 0 or self.f_weight != 0:
            loss_ic, loss_f = self.nsloss(predict, x)
        else:
            loss_ic, loss_f = mnp.zeros((1)), mnp.zeros((1))

        total_loss = loss_l2 * self.xy_weight + loss_f * \
            self.f_weight + loss_ic * self.ic_weight
        return total_loss


class NSTrainEqnLoss(nn.Cell):
    def __init__(
            self,
            net,
            nsloss,
            batch_size,
            f_weight,
            ic_weight,
            nx_eqn,
            nt_eqn,
            *args,
            **kwargs):
        super(NSTrainEqnLoss, self).__init__()
        self.net = net
        self.batchsize = batch_size
        self.f_weight = f_weight
        self.ic_weight = ic_weight
        self.reshape = ms.ops.Reshape()
        self.nsloss = nsloss
        self.nx = nx_eqn
        self.nt = nt_eqn
        self.zeros = ms.ops.Zeros()

    def construct(self, x):
        padding = self.zeros(
            (x.shape[0], x.shape[1], x.shape[-3], 5, x.shape[-1]), ms.float32)
        x_in = ms.ops.Concat(-2)((x.astype(padding.dtype), padding))
        predict = self.net(x_in)
        predict = self.reshape(
            predict, (self.batchsize, self.nx, self.nx, self.nt + 5))
        predict = predict[..., :-
                          5].reshape(self.batchsize, self.nx, self.nx, self.nt)
        x = x[:, :, :, 0, -1]

        loss_ic, loss_f = self.nsloss(predict, x)

        total_loss = loss_f * self.f_weight + loss_ic * self.ic_weight
        return total_loss


class NSTestDataLoss(nn.Cell):
    def __init__(self, net, lploss, nx_data, nt_data, batch_size, *args, **kwargs):
        super(NSTestDataLoss, self).__init__()
        self.net = net
        self.batchsize = batch_size
        self.reshape = ms.ops.Reshape()
        self.lploss = lploss
        self.nx = nx_data
        self.nt = nt_data

    def construct(self, x, y):
        padding_shape = (x.shape[0], x.shape[1], x.shape[-3], 5, x.shape[-1])
        padding = mnp.zeros(padding_shape)
        x_in = ms.ops.Concat(-2)((x.astype(padding.dtype), padding))
        predict = self.net(x_in)
        predict = self.reshape(
            predict, (self.batchsize, self.nx, self.nx, self.nt + 5))
        predict = predict[..., :-
                          5].reshape(self.batchsize, self.nx, self.nx, self.nt)
        y = y.reshape(self.batchsize, self.nx, self.nx, self.nt)

        loss_l2 = self.lploss(predict, y)

        return loss_l2


class NSTestEqnLoss(nn.Cell):
    def __init__(self, net, nsloss, batch_size, nx_eqn, nt_eqn, *args, **kwargs):
        super(NSTestEqnLoss, self).__init__()
        self.net = net
        self.batchsize = batch_size
        self.reshape = ms.ops.Reshape()
        self.nsloss = nsloss
        self.nx = nx_eqn
        self.nt = nt_eqn
        self.zeros = ms.ops.Zeros()

    def construct(self, x, y):
        padding = self.zeros(
            (x.shape[0], x.shape[1], x.shape[-3], 5, x.shape[-1]), ms.float32)
        x_in = ms.ops.Concat(-2)((x.astype(padding.dtype), padding))
        predict = self.net(x_in)
        predict = self.reshape(
            predict, (self.batchsize, self.nx, self.nx, self.nt + 5))
        predict = self.reshape(
            predict[..., :-5], (self.batchsize, self.nx, self.nx, self.nt))
        x = x[:, :, :, 0, -1]

        _, loss_f = self.nsloss(predict, x)

        return loss_f
