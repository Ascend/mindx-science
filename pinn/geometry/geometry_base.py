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
from mindelec.geometry.geometry_base import Geometry as GEO
from mindelec.geometry.geometry_base import PartSamplingConfig as PSC
from mindelec.geometry.geometry_base import SamplingConfig as SC


class Geometry(GEO):
    def __init__(self, name, dim, coord_min, coord_max, dtype=np.float32, sampling_config=None):
        super(Geometry, self).__init__(name, dim, coord_min, coord_max, dtype, sampling_config)


class PartSamplingConfig(PSC):
    def __init__(self, size, random_sampling=True, sampler="uniform", random_merge=True, with_normal=False):
        super(PartSamplingConfig, self).__init__(size, random_sampling, sampler, random_merge, with_normal)


class SamplingConfig(SC):
    def __init__(self, part_sampling_dict):
        super(SamplingConfig, self).__init__(part_sampling_dict)
