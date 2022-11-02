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

from mindelec.solver import Solver as S


class Solver(S):
    def __init__(self, network, optimizer, loss_fn="l2", mode="Data", train_constraints=None, test_constraints=None,
                 train_input_map=None, test_input_map=None, mtl_weighted_cell=None, latent_vector=None, latent_reg=1e-2,
                 metrics=None, eval_network=None,
                 eval_indexes=None, amp_level="O0", **kwargs):
        super(Solver, self).__init__(network, optimizer, loss_fn, mode, train_constraints, test_constraints,
                                     train_input_map, test_input_map, mtl_weighted_cell, latent_vector, latent_reg,
                                     metrics, eval_network, eval_indexes, amp_level, **kwargs)
