# !usr/bin/env python
# -*- coding:utf-8 _*-

from mindelec.data import Dataset as ds


class Dataset(ds):
    def __init__(self, geometry_dict=None, existed_data_list=None, dataset_list=None):
        super(Dataset, self).__init__(geometry_dict, existed_data_list, dataset_list)