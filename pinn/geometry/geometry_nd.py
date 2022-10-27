# !usr/bin/env python
# -*- coding:utf-8 _*-

import numpy as np
from mindelec.geometry.geometry_1d import Interval as In
from mindelec.geometry.geometry_2d import Disk as D
from mindelec.geometry.geometry_2d import Rectangle as R
from mindelec.geometry.geometry_3d import Cuboid as C
from mindelec.geometry.geometry_nd import HyperCube as H

class Interval(In):
    def __init__(self, name, coord_min, coord_max, dtype=np.float32, sampling_config=None):
        super(Interval, self).__init__(name, coord_min, coord_max, dtype, sampling_config)


class Disk(D):
    def __init__(self, name, center, radius, dtype=np.float32, sampling_config=None):
        super(Disk, self).__init__(name, center, radius, dtype, sampling_config)


class Rectangle(R):
    def __init__(self,name, coord_min, coord_max, dtype=np.float32, sampling_config=None):
        super(Rectangle, self).__init__(name, coord_min, coord_max, dtype, sampling_config)


class Cuboid(C):
    def __init__(self ,name, coord_min, coord_max, dtype=np.float32, sampling_config=None):
        super(Cuboid, self).__init__(name, coord_min, coord_max, dtype, sampling_config)


class HyperCube(H):
    def __init__(self, name, dim, coord_min, coord_max, dtype=np.float32, sampling_config=None):
        super(HyperCube, self).__init__(name, dim, coord_min, coord_max, dtype, sampling_config)