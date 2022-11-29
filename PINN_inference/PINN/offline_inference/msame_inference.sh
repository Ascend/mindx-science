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

