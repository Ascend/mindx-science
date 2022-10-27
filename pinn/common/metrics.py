# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:Marcel Tuan
@Blog: https://github.com/AlfredMarcel
 
@File: metrics.py.py
@Time: 2022/10/19 9:39
"""

from mindelec.common import L2 as L2metric

class L2(L2metric):
    def __init__(self):
        super(L2, self).__init__()