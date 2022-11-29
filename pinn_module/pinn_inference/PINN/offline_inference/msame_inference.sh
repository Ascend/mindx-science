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
. /usr/local/Ascend/ascend-toolkit/set_env.sh &&

./msame.aarch64 --model "./PINN/offline_inference/Poisson.om" \
--input "./data/Poisson.bin" \
--output "./result" --outfmt TXT

./msame.aarch64 --model "./PINN/offline_inference/Schrodinger.om" \
--input "./data/Schrodinger.bin" \
--output "./result" --outfmt TXT

./msame.aarch64 --model "./PINN/offline_inference/NS.om" \
--input "./data/NS.bin" \
--output "./result" --outfmt TXT

