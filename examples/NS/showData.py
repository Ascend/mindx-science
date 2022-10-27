# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:Marcel Tuan
@Blog: https://github.com/AlfredMarcel
 
@File: showData.py
@Time: 2022/9/3 13:25
"""

"""
=========================用于检查输入数据的脚本=========================
"""

import json
import numpy as np
import sys

configs = json.load(open("./config.json"))
for filename in configs["train_data_path"]:
    arr = np.load(filename)
    print(arr.shape)
    print("---------------------" + filename + "----------------")
    np.set_printoptions(threshold=sys.maxsize)
    print(arr)
    print(type(arr[0][0]))
