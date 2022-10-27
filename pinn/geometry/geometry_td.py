# !usr/bin/env python
# -*- coding:utf-8 _*-

import numpy as np
from mindelec.geometry.geometry_td import TimeDomain as TD
from mindelec.geometry.geometry_td import GeometryWithTime as GWT


class TimeDomain(TD):
    def __init__(self, name, start=0.0, end=1.0, dtype=np.float32, sampling_config=None):
        super(TimeDomain, self).__init__(name, start, end, dtype, sampling_config)


class GeometryWithTime(GWT):
    def __init__(self,geometry, timedomain, sampling_config=None):
        super(GeometryWithTime, self).__init__(geometry, timedomain, sampling_config)
