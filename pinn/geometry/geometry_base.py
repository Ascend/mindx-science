# !usr/bin/env python
# -*- coding:utf-8 _*-

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
    def __init__(self,part_sampling_dict):
        super(SamplingConfig, self).__init__(part_sampling_dict)
