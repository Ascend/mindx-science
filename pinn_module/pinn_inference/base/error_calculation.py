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


def poisson_error(results, labels):
    delta = results - labels
    return np.sqrt(np.sum(np.square(delta))) / np.sqrt(np.sum(np.square(labels)))


def schrodinger_error(results, labels):
    h_results = np.sqrt(np.square(results[:, 0]) + np.square(results[:, 1]))
    h_labels = np.sqrt(np.square(labels[:, 0]) + np.square(labels[:, 1]))
    h_error = h_labels - h_results
    return np.sqrt(np.sum(np.square(h_error))) / np.sqrt(np.sum(np.square(h_labels)))


def ns_error(results, labels):
    delta = results - labels
    l2_u = np.sqrt(np.sum(np.square(delta[:, 0]))) / np.sqrt(np.sum(np.square(labels[:, 0])))
    l2_v = np.sqrt(np.sum(np.square(delta[:, 1]))) / np.sqrt(np.sum(np.square(labels[:, 1])))
    return l2_u, l2_v
