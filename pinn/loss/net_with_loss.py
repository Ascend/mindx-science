# !usr/bin/env python
# -*- coding:utf-8 _*-

from mindelec.loss import NetWithLoss as NWL
from mindelec.loss import NetWithEval as NWE

class NetWithLoss(NWL):
    def __init__(self,net_without_loss, constraints, loss="l2", dataset_input_map=None,
                 mtl_weighted_cell=None, latent_vector=None, latent_reg=0.01):
        super(NetWithLoss, self).__init__(net_without_loss, constraints, loss, dataset_input_map, mtl_weighted_cell, latent_vector, latent_reg)


class NetWithEval(NWE):
    def __init__(self,net_without_loss, constraints, loss="l2", dataset_input_map=None):
        super(NetWithEval, self).__init__(net_without_loss, constraints, loss, dataset_input_map)
