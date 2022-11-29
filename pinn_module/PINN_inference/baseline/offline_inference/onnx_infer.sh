trtexec --onnx=./baseline/offline_inference/Poisson_deepxde.onnx
trtexec --onnx=./baseline/offline_inference/Schrodinger_deepxde.onnx --fp16
trtexec --onnx=./baseline/offline_inference/NS_deepxde.onnx