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


from .geometry_base import Geometry, PartSamplingConfig, SamplingConfig
from .geometry_nd import Interval,Disk,Rectangle, Cuboid
from .geometry_nd import HyperCube
from .geometry_td import TimeDomain, GeometryWithTime
from .polygon import Polygon
from .csg import CSGIntersection, CSGDifference, CSGUnion, CSGXOR
from mindelec.geometry.utils import create_config_from_edict

__all__ = [
    "Geometry",
    "PartSamplingConfig",
    "SamplingConfig",
    "Interval",
    "Disk",
    "Rectangle",
    "Cuboid",
    "HyperCube",
    "TimeDomain",
    "GeometryWithTime",
    "CSGIntersection",
    "CSGDifference",
    "CSGUnion",
    "CSGXOR",
    "Polygon",
    "create_config_from_edict"
]