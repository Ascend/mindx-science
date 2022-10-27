# !usr/bin/env python
# -*- coding:utf-8 _*-

import numpy as np
import mindspore

from mindelec.operators import Grad as G
from mindelec.operators import  SecondOrderGrad as SOG

class Grad(G):
    def __init__(self, model, argnum=0):
        super(Grad, self).__init__(model, argnum)

class SecondOrderGrad(SOG):
    def __init__(self, model, input_idx1, input_idx2, output_idx):
        super(SecondOrderGrad, self).__init__(model, input_idx1, input_idx2, output_idx)