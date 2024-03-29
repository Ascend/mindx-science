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
# ==============================================================================
"""sampling information"""
from easydict import EasyDict as edict


polygon_sampling_config = edict({
    'domain': edict({
        'random_sampling': True,
        'size': 3000,
        'sampler': 'uniform'
    }),
    'BC': edict({
        'random_sampling': True,
        'size': 751,
        'sampler': 'uniform',
        'with_normal': False
    })
})
