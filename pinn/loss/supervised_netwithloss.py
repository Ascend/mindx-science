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

import mindspore

from mindelec.loss import NetWithLoss
import numpy as np
from mindspore import log as logger
from mindspore import nn
from mindspore import ops
from mindspore import Tensor, Parameter

def _transfer_tensor_to_tuple(inputs):
    """
    If the input is a tensor, convert it to a tuple. If not, the output is unchanged.
    """
    if isinstance(inputs, Tensor):
        return (inputs,)

    return inputs


class SupervisedNetWithLoss(NetWithLoss):
    def __init__(self, net_without_loss, constraints, loss="l2", dataset_input_map=None,
                 mtl_weighted_cell=None, latent_vector=None, latent_reg=0.01):
        super(SupervisedNetWithLoss, self).__init__(net_without_loss,constraints,loss,dataset_input_map,mtl_weighted_cell,latent_vector,latent_reg)
        self.label_mark = constraints.label_data

    def construct(self, *inputs):
        data = _transfer_tensor_to_tuple(inputs)
        loss = {}
        total_loss = self.zero
        for name in self.dataset_columns_map.keys():
            if not self.judgeLabel(name):
                columns_list = self.dataset_columns_map[name]
                input_data = {}
                for column_name in columns_list:
                    input_data[column_name] = data[self.column_index_map[column_name]]
                net_input = ()
                for column_name in self.dataset_input_map[name]:
                    net_input += (data[self.column_index_map[column_name]],)
                out = self.net_without_loss(*net_input)
                out = _transfer_tensor_to_tuple(out)
                base = self.fn_cell_list[self.dataset_cell_index_map[name]](*out, **input_data)
                temp_loss = self.reduce_mean(self.loss_fn_dict[name](base, self.zeros_like(base)))
                loss[name] = temp_loss
                total_loss += temp_loss
            else:
                "------------------------------------------------------------------------"
                "传入的是有监督数据时,分离最后一个维度的数据（输出），计算损失"
                columns_list = self.dataset_columns_map[name]
                idx = self.label_mark[name]
                input_data = {}
                label_data = {}
                for column_name in columns_list:
                    input_data[column_name] = data[self.column_index_map[column_name]][...,:idx]
                    label_data = data[self.column_index_map[column_name]][...,idx:]
                net_input = ()
                for column_name in self.dataset_input_map[name]:
                    net_input += (data[self.column_index_map[column_name]][...,:idx],)
                out = self.net_without_loss(*net_input)
                base = out[...,0:2]
                temp_loss = self.reduce_mean(self.loss_fn_dict[name](base, label_data))
                loss[name] = temp_loss
                total_loss += temp_loss

        if self.mtl_cell is not None:
            total_loss = self.mtl_cell(loss.values())

        if self.latent_vector is not None:
            loss_reg = self.latent_reg * self.reduce_mean(self.pow(self.latent_vector, 2))
            loss["reg"] = loss_reg
            total_loss += loss_reg

        loss["total_loss"] = total_loss

        return total_loss

    def judgeLabel(self,dataset_name):
        '''
        Args:
            dataset_name:
        Returns:
            若dataset_name为带label数据，返回true
        '''
        for i in self.label_mark.keys():
            if dataset_name == i:
                return True
        return False