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