# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
create dataset
"""
import numpy as np
import copy
from pinn.data import Dataset, ExistedDataConfig
from pinn.geometry import Rectangle, Geometry, CSGDifference, create_config_from_edict
from pinn.geometry import Polygon

from .sampling_config import *

def get_test_data(test_data_path):
    """load labeled data for evaluation"""
    data = np.load(test_data_path)
    inputs = data['X_test']
    label = data['y_ref']
    return inputs, label

def create_random_dataset(config):
    if config["method"] == "difference":
        outer_domain = Rectangle("rect1", config["coord_min"], config["coord_max"])
        in_domain = Rectangle("rect2", config["coord_mid"], config["coord_max"])
        inside_domain = CSGDifference(outer_domain, in_domain)
    elif config["method"] == "polygon":
        vertex_list = config["vertex_list"]
        inside_domain = Polygon("in_src", vertex_list)
    inside_domain.set_name("in_src")
    inside_domain.set_sampling_config(create_config_from_edict(polygon_sampling_config))
    geom_dict = {inside_domain: ["domain", "BC"]}
    dataset = Dataset(geom_dict)
    return dataset
