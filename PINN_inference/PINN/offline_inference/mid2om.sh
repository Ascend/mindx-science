#!/bin/bash
source ./PINN/offline_inference/env_npu.sh
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --framework=5 --model=./PINN/offline_inference/Poisson_pinn.onnx \
--input_shape="x:19521,2"  --output=./PINN/offline_inference/Poisson --soc_version=Ascend310 --precision_mode=allow_mix_precision

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --framework=5 --model=./PINN/offline_inference/Schrodinger_pinn.onnx \
--input_shape="x:51456,2"  --output=./PINN/offline_inference/Schrodinger --soc_version=Ascend310 --precision_mode=allow_mix_precision

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --framework=5 --model=./PINN/offline_inference/NS_pinn.onnx \
--input_shape="x:355000,3"  --output=./PINN/offline_inference/NS --soc_version=Ascend310 --precision_mode=allow_mix_precision