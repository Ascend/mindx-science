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

# class SupervisedDataset(Dataset):
#     def __init__(self, geometry_dict=None, existed_data_list=None, dataset_list=None, label_data_list=None):
#         # 在mindelec.data.dataset 基础上添加标签
#         val = [i.name for i in existed_data_list]
#         label = [i.name for i in label_data_list]
#         self.supervised_dict = dict(zip(val,label))
#         l1 = [i.name for i in existed_data_list] if existed_data_list is not None else []
#         l2 = [i.name for i in geometry_dict] if geometry_dict is not None else []
#         l3 = [i.name for i in dataset_list] if dataset_list is not None else []
#         self.train_data_list = l1+l2+l3
#
#         existed_data_list=existed_data_list+label_data_list
#         super(SupervisedDataset, self).__init__(geometry_dict, existed_data_list, dataset_list)
#
#     def _get_all_datasets(self):
#         """get all datasets"""
#         if self.geometry_dict is not None:
#             for geom, types in self.geometry_dict.items():
#                 for geom_type in types:
#                     dataset = self._create_dataset_from_geometry(geom, geom_type)
#                     self.all_datasets.append(dataset)
#
#         if self.existed_data_list is not None:
#             for data_config in self.existed_data_list:
#                 dataset = ExistedDataset(data_config=data_config)
#                 self.all_datasets.append(dataset)



class SupervisedDataset(Dataset):
    def __init__(self, geometry_dict=None, existed_data_list=None, dataset_list=None, label_data=None):
        super(SupervisedDataset, self).__init__(geometry_dict, existed_data_list, dataset_list)
        self.label_data = label_data