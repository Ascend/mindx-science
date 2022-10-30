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

from mindelec.loss import Constraints
#
# class SupervisedConstraints(Constraints):
#     def __init__(self, dataset, pde_dict):
#         super(SupervisedConstraints, self).__init__(dataset, pde_dict)
#         self.supervised_dict = dataset.supervised_dict
#         self.train_data_list = dataset.train_data_list


class SupervisedConstraints(Constraints):
    def __init__(self, dataset, pde_dict):
        super(SupervisedConstraints, self).__init__(dataset, pde_dict)
        self.label_data = dataset.label_data


