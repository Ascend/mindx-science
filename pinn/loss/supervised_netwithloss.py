# !usr/bin/env python
# -*- coding:utf-8 _*-
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

# class SupervisedNetWithLoss(NetWithLoss):
#     def __init__(self, net_without_loss, constraints, loss="l2", dataset_input_map=None,
#                  mtl_weighted_cell=None, latent_vector=None, latent_reg=0.01):
#         super(SupervisedNetWithLoss, self).__init__(net_without_loss,constraints,loss,dataset_input_map,mtl_weighted_cell,latent_vector,latent_reg)
#         self.supervised_dict = constraints.supervised_dict
#         self.train_data_list = constraints.train_data_list
#
#     def construct(self, *inputs):
#         data = _transfer_tensor_to_tuple(inputs)
#         loss = {}
#         total_loss = self.zero
#         for name in self.dataset_columns_map.keys():
#             if self.isTrainDataset(name):
#                 if not self.judgeLabel(name):
#                     columns_list = self.dataset_columns_map[name]
#                     input_data = {}
#                     for column_name in columns_list:
#                         input_data[column_name] = data[self.column_index_map[column_name]]
#                     net_input = ()
#                     for column_name in self.dataset_input_map[name]:
#                         net_input += (data[self.column_index_map[column_name]],)
#                     out = self.net_without_loss(*net_input)
#                     out = _transfer_tensor_to_tuple(out)
#                     base = self.fn_cell_list[self.dataset_cell_index_map[name]](*out, **input_data)
#                     temp_loss = self.reduce_mean(self.loss_fn_dict[name](base, self.zeros_like(base)))
#                     loss[name] = temp_loss
#                     total_loss += temp_loss
#                 else:
#                     "------------------------------------------------------------------------"
#                     "传入的是有监督数据"
#                     columns_list = self.dataset_columns_map[name]
#                     l_name = self.supervised_dict[name]
#                     label_list = self.dataset_columns_map[l_name]
#                     input_data = {}
#                     label_data = {}
#                     for column_name in columns_list:
#                         input_data[column_name] = data[self.column_index_map[column_name]]
#                     for label_name in label_list:
#                         label_data[label_name] = data[self.column_index_map[label_name]]
#                     net_input = ()
#                     for column_name in self.dataset_input_map[name]:
#                         net_input += (data[self.column_index_map[column_name]],)
#                     out = self.net_without_loss(*net_input)
#                     #out = _transfer_tensor_to_tuple(out)
#                     base = out[...,3:5]
#                     temp_loss = self.reduce_mean(self.loss_fn_dict[name](base, label_data))
#                     loss[name] = temp_loss
#                     total_loss += temp_loss
#
#         if self.mtl_cell is not None:
#             total_loss = self.mtl_cell(loss.values())
#
#         if self.latent_vector is not None:
#             loss_reg = self.latent_reg * self.reduce_mean(self.pow(self.latent_vector, 2))
#             loss["reg"] = loss_reg
#             total_loss += loss_reg
#
#         loss["total_loss"] = total_loss
#
#         return total_loss
#     #
#     def judgeLabel(self,name):
#         for i in self.supervised_dict.keys():
#             if name == i:
#                 return True
#         # if self.supervised_dict.get(name,-1)==-1:
#         #     return True
#         return False
#
#     def isTrainDataset(self,name):
#         for i in self.train_data_list:
#             if name == i:
#                 return True
#         return False

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
                #out = _transfer_tensor_to_tuple(out)
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
        # "------------------debug用，查看各部分loss--------------------"

        # loss_re = []
        # for i in loss.keys():
        #     loss_re.append(loss[i])
        # return loss_re[0]

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