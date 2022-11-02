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
from mindelec.geometry.geometry_td import TimeDomain as TD
from mindelec.geometry.geometry_td import GeometryWithTime as GWT


class TimeDomain(TD):
    def __init__(self, name, start=0.0, end=1.0, dtype=np.float32, sampling_config=None):
        super(TimeDomain, self).__init__(name, start, end, dtype, sampling_config)


class GeometryWithTime(GWT):
    def __init__(self, geometry, timedomain, sampling_config=None):
        super(GeometryWithTime, self).__init__(geometry, timedomain, sampling_config)
