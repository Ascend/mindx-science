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

python -m onnxsim --overwrite-input-shape="19521,2" ./PINN/offline_inference/Poisson_pinn.onnx ./PINN/offline_inference/Poisson_pinn.onnx
python -m onnxsim --overwrite-input-shape="51456,2" ./PINN/offline_inference/Schrodinger_pinn.onnx ./PINN/offline_inference/Schrodinger_pinn.onnx
python -m onnxsim --overwrite-input-shape="355000,3" ./PINN/offline_inference/NS_pinn.onnx ./PINN/offline_inference/NS_pinn.onnx