# !usr/bin/env python
# -*- coding:utf-8 _*-

from mindelec.solver import Solver
from pinn.loss import SupervisedNetWithLoss
from mindspore.train import amp
from mindelec.loss import NetWithLoss

class SupervisedSolver(Solver):
    def __init__(self, network, optimizer, loss_fn="l2", mode="Data", train_constraints=None, test_constraints=None,
                 train_input_map=None, test_input_map=None, mtl_weighted_cell=None, latent_vector=None, latent_reg=1e-2,
                 metrics=None, eval_network=None,
                 eval_indexes=None, amp_level="O0", **kwargs):
        super(SupervisedSolver, self).__init__(network, optimizer, loss_fn, mode, train_constraints, test_constraints,
                 train_input_map, test_input_map, mtl_weighted_cell, latent_vector, latent_reg,metrics, eval_network,
                 eval_indexes, amp_level, **kwargs)

    def _build_train_network(self):
        """Build train network"""
        loss_network = SupervisedNetWithLoss(self._network, self._train_constraints, self._loss_fn,
                                   dataset_input_map=self._train_input_map,
                                   mtl_weighted_cell=self.mtl_weighted_cell,
                                   latent_vector=self.latent_vector,
                                   latent_reg=self.latent_reg)
        if self._loss_scale_manager_set:
            network = amp.build_train_network(loss_network,
                                              self._optimizer,
                                              level=self._amp_level,
                                              loss_scale_manager=None,
                                              keep_batchnorm_fp32=self._keep_bn_fp32)
        else:
            network = amp.build_train_network(loss_network,
                                              self._optimizer,
                                              level=self._amp_level,
                                              keep_batchnorm_fp32=self._keep_bn_fp32)
        return network
