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

from mindelec.loss import NetWithLoss as NWL
from mindelec.loss import NetWithEval as NWE


class NetWithLoss(NWL):
    def __init__(self, net_without_loss, constraints, loss="l2", dataset_input_map=None,
                 mtl_weighted_cell=None, latent_vector=None, latent_reg=0.01):
        super(NetWithLoss, self).__init__(net_without_loss, constraints, loss,
                                          dataset_input_map, mtl_weighted_cell, latent_vector, latent_reg)


class NetWithEval(NWE):
    def __init__(self, net_without_loss, constraints, loss="l2", dataset_input_map=None):
        super(NetWithEval, self).__init__(net_without_loss, constraints, loss, dataset_input_map)
