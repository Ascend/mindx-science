#!/bin/bash

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

source ./PINN/offline_inference/env_npu.sh
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --framework=5 --model=./PINN/offline_inference/Poisson_pinn.onnx \
--input_shape="x:19521,2"  --output=./PINN/offline_inference/Poisson --soc_version=Ascend310 --precision_mode=allow_mix_precision

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --framework=5 --model=./PINN/offline_inference/Schrodinger_pinn.onnx \
--input_shape="x:51456,2"  --output=./PINN/offline_inference/Schrodinger --soc_version=Ascend310 --precision_mode=allow_mix_precision

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --framework=5 --model=./PINN/offline_inference/NS_pinn.onnx \
--input_shape="x:355000,3"  --output=./PINN/offline_inference/NS --soc_version=Ascend310 --precision_mode=allow_mix_precision