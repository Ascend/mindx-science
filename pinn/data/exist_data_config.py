# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:Marcel Tuan
@Blog: https://github.com/AlfredMarcel
 
@File: exist_data_config.py
@Time: 2022/10/19 10:11
"""

from mindelec.data import ExistedDataConfig as EDC

class ExistedDataConfig(EDC):
    def __init__(self,name, data_dir, columns_list, data_format="npy", constraint_type="Label", random_merge=True):
        super(ExistedDataConfig, self).__init__(name, data_dir, columns_list, data_format, constraint_type, random_merge)
