# !usr/bin/env python
# -*- coding:utf-8 _*-

from mindelec.geometry import CSGDifference as CSGD
from mindelec.geometry import CSGIntersection as CSGI
from mindelec.geometry import CSGXOR as CSGX
from mindelec.geometry import CSGUnion as CSGU


class CSGDifference(CSGD):
    def __init__(self, geom1, geom2, sampling_config=None):
        super(CSGDifference, self).__init__(geom1, geom2, sampling_config)


class CSGIntersection(CSGI):
    def __init__(self, geom1, geom2, sampling_config=None):
        super(CSGIntersection, self).__init__(geom1, geom2, sampling_config)


class CSGXOR(CSGX):
    def __init__(self, geom1, geom2, sampling_config=None):
        super(CSGXOR, self).__init__(geom1, geom2, sampling_config)


class CSGUnion(CSGU):
    def __init__(self, geom1, geom2, sampling_config=None):
        super(CSGUnion, self).__init__(geom1, geom2, sampling_config)