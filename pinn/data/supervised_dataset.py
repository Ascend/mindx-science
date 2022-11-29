'''
Description: 说明
Author: Marcel
Date: 2022-11-01 12:44:07
'''
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

from mindelec.data import Dataset
from mindelec.data.existed_data import ExistedDataset


class SupervisedDataset(Dataset):
    def __init__(self, geometry_dict=None, existed_data_list=None, dataset_list=None, label_data=None):
        super(SupervisedDataset, self).__init__(geometry_dict, existed_data_list, dataset_list)
        self.label_data = label_data