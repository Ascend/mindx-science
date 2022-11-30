# PINN for inference

## 1 介绍
pinn组件实现了求解正/逆偏微分方程的模块化通用工具包。其中通用模块包括：网络结构定义、数据采样生成、区域定义、损失函数生成及数据匹配、算子定义、求解器定义等。pinn组件还一并实现了求解含非线性项的一维含时Schrödinger方程、二维L型区域内的泊松方程、Navier-Stokes方程逆问题这三个偏微分方程，以作为参考样例。

## 2 软件架构
| 主要依赖包                    | 版本              |
|--------------------------|-----------------
| <center>mindspore-ascend | <center>1.7.0   |
| <center>mindscience-mindelec-ascend       | <center>0.1.0rc1  |



## 3 环境准备

### 3.1 虚拟环境
对于基线复现，需要使用配置文件创建虚拟环境
```
conda env create -f ./baseline.yaml
```
对于PINN复现，需要设置环境变量
```
. /usr/local/Ascend/ascend-toolkit/set_env.sh
```
### 3.2 第三方依赖包
| 主要依赖包                               | 版本                |
|-------------------------------------|-------------------
| <center>mindspore-ascend            | <center>1.7.0     |
| <center>mindscience-mindelec-ascend | <center>0.1.0 rc1 |
| <center>mindx                       | <center>3.0 rc3   |
| <center>numpy                       | <center>1.22.3    |
| <center>scipy                       | <center>1.9.3     |
| <center>scikit-learn                       | <center>1.0.2     |

### 3.3 数据准备
- 数据集<br>
在PINN_inference下创建data文件夹，在文件夹存放三个方程所用到的数据集。<br>
对于泊松方程，从此[链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindx_science/pinns/Poisson_Lshape_clean.npz) 下载；<br>
对于薛定谔方程，从此[链接](http://mindx.sdk.obs.cn-snorth-4.myhuaweicloud.com/mindx_science/pinns/NLS.mat) 下载；<br>
对于NS方程，从此[链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindx_science/pinns/cylinder_nektar_wake.mat) 下载；<br>


- 权重文件 <br>
在PINN_inference下创建pretrained_models文件夹，在文件夹存放三个方程所用到的数据集。<br>
三个方程所需要的权重文件从此[链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindx_science/pinns/pinn_ckpt.7z) 下载

## 4 推理说明

首先请配置环境变量。将pinn_inference添加至python环境变量。

    export PYTHONPATH=/{your path}/pinn_inference


### 4.1 baseline推理
#### 4.1.1 在线推理
```
python ./baseline/online_inference.py
```
#### 4.1.2 离线推理
(1) pth文件转onnx文件
```
python ./baseline/offline_inference/pth2onnx.py
```

(2) onnx文件在T4服务器上经过TensorRT进行推理，获得性能数据
```
bash ./baseline/offline_inference/onnx_infer.sh
```

### 4.2 PINN推理
#### 4.2.1 在线推理
```
python ./PINN/online_inference.py
```
#### 4.2.2 离线推理
(1) 在训练服务器和推理服务器上设置环境变量
```
. /usr/local/Ascend/ascend-toolkit/set_env.sh
```
(2) 在训练服务器（已安装mindspore库）上将pth文件转中间文件
 
- 首先修改config.json的"mid_file_format"参数，设置需要输出的文件格式，包括"ONNX", "AIR"；

- 然后执行以下命令：
```
python ./PINN/offline_inference/ckpt2mid.py
```

- 若中间文件是ONNX文件，则需要使用onnxsim简化
```
bash ./PINN/offline_inference/onnx_sim.sh
```
- 最后将生成的中间文件传输到对应的推理服务器上

(3) 在推理服务器上，使用ATC工具将中间文件转换为om文件
```
bash ./PINN/offline_inference/mid2om.sh
```
注意：当中间文件为AIR文件时，--framwork设置为1；当中间文件为ONNX文件时，--framwork设置为5；--input_shape根据batch_size而设定

(4) 数据前处理
```
python ./PINN/offline_inference/data_preprocess.py
```
(5) msame推理，得到性能数据
```
bash ./PINN/offline_inference/msame_inference.sh
```
(6) 数据后处理，得到精度数据
```
# 首先需要把三个推理结果的txt文件移动到result目录下
python ./PINN/offline_inference/data_postprocess.py
```
### 4.3 SDK推理
(1) 设置环境变量
```
. /usr/local/Ascend/ascend-toolkit/set_env.sh
. $mxVision-3.0.RC3_path$/set_env.sh
```
(2) sdk推理，得到性能和精度数据
```
python3.9 ./sdk/om_infer.py
```
## 5 结果说明
### 5.1 在线推理
环境说明：
GPU服务器
NPU服务器


| 方程         | GPU推理精度↓    | NPU推理精度↓                   | GPU推理性能↓      |NPU推理性能↓
|---------------|------|----------------------------|---------------|----  
| <center>Poisson   | <center>0.1288  | <center>0.1043             | <center>29.8  |<center>0.7724
| <center>Schrodinger | <center>0.0017 | <center>0.001686           | <center>212.5 |<center>0.9691
| <center>NS    | <center>0.0074, 0.0251 | <center> 0.004746, 0.01817 | <center>525.6  |<center>3.5256

注：\
（1）GPU推理的设备条件是单张NVIDIA V100 16GB显存GPU；NPU推理的设备条件是单张910A昇腾芯片\
（2）推理精度计算公式为L2范数的相对误差，该指标越低代表推理越准确\
（3）推理性能的单位是毫秒
### 5.2 离线推理
| 方程         | ckpt权重文件精度↓                   | NPU推理精度↓                      | 推理误差损失            |GPU推理性能↑  |NPU推理性能↑
|--------------------------|-------------------------------|-------------------------------|-------------------|---- | ---|
| <center>Poisson    | <center>0.1043                | <center>0.1044                | <center>0.1%      |<center>0.42    |<center>1.40
| <center>Schrodinger | <center>0.001686              | <center> 0.001925             | <center>14.17%    |<center>1.67    |<center>4.13
| <center>NS       | <center>0.00474582, 0.0181725 | <center>0.00481012, 0.0183875 | <center>1.3%,1.1% |<center>61.6339 |<center>28.003
注：\
（1）GPU推理的设备条件是单张NVIDIA V100 16GB显存GPU；NPU推理的设备条件是单张910A昇腾芯片\
（2）推理精度计算公式为L2范数的相对误差，该指标越低代表推理越准确\
（3）NPU推理精度工具为sdk，性能测试工具为msame\
（3）推理误差损失计算公式=|NPU推理精度-ckpt推理精度|/ckpt权重文件精度 *100% \
（4）推理性能的单位是毫秒

