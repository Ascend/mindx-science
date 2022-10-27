# !usr/bin/env python
# -*- coding:utf-8 _*-

from mindelec.solver import Solver as S

class Solver(S):
    def __init__(self,network, optimizer, loss_fn="l2", mode="Data", train_constraints=None, test_constraints=None,
                 train_input_map=None, test_input_map=None, mtl_weighted_cell=None, latent_vector=None, latent_reg=1e-2,
                 metrics=None, eval_network=None,
                 eval_indexes=None, amp_level="O0", **kwargs):
        super(Solver, self).__init__(network, optimizer, loss_fn, mode, train_constraints, test_constraints,
                 train_input_map, test_input_map, mtl_weighted_cell, latent_vector, latent_reg,
                 metrics, eval_network, eval_indexes, amp_level, **kwargs)
