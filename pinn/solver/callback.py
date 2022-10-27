# !usr/bin/env python
# -*- coding:utf-8 _*-

from mindspore.train.callback import Callback
from mindelec.solver import LossAndTimeMonitor as LT

class LossAndTimeMonitor(LT):
    def __init__(self, data_size, per_print_times=1):
        super(LossAndTimeMonitor, self).__init__(data_size,per_print_times)

