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