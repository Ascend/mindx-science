# PINN工具包

#### 介绍
pinn组件实现了求解正/逆偏微分方程的模块化通用工具包。其中通用模块包括：网络结构定义、数据采样生成、区域定义、损失函数生成及数据匹配、算子定义、求解器定义等。pinn组件还一并实现了求解含非线性项的一维含时Schrödinger方程、二维L型区域内的泊松方程、Navier-Stokes方程逆问题这三个偏微分方程，以作为参考样例

#### 使用说明

首先请配置环境变量,将pinn包根目录添加至python环境变量。

    export PYTHONPATH=/{your path}/pinn_module

- NS方程
1. clone本仓库到本地
2. 从此[链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindx_science/pinns/cylinder_nektar_wake.mat) 下载数据，存放到examples/NS下。
3. 执行examples/NS/handleData.py ,随机采样生成数据集（data_input.npy,data_uv.npy）。
4. 配置config.json中的设备，训练轮数等。
5. 执行train.py

- Possion方程
1. clone本仓库到本地
2. 从此[链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindx_science/pinns/Poisson_Lshape_clean.npz) 下载数据，存放到examples/Possion_Lshape下。
3. 配置config.json中的设备，训练轮数等。
4. 执行train.py

- Schrodinger方程
1. clone本仓库到本地
2. 从此[链接](http://mindx.sdk.obs.cn-snorth-4.myhuaweicloud.com/mindx_science/pinns/NLS.mat) 下载数据，存放到examples/Schrodinger下。
3. 配置config.json中的设备，训练轮数等。
4. 执行train.py（执行策略为：第一次执行学习率为0.001，第二次执行学习率为0.0001.两次执行的milestones和epoch_num均为config默认）


#### 软件架构
包含使用样例examples与工具包pinn两个部分。目录树如下：

    ├─examples
    │  ├─NS
    │  │  └─src
    │  ├─Possion_Lshape
    │  │  └─src
    │  └─Schrodinger
    │      └─src
    └─pinn
        ├─architecture
        ├─common
        ├─data
        ├─geometry
        ├─loss
        ├─operators
        └─solver

在pinn工具包之中：
-	Architecture模块：网络结构定义，通用的MLP结构以及自定义网络结构。
-	Common模块：学习率调整以及策略定义。
-	Data模块：涉及数据集的创建，数据采样过程，有监督数据的读入、划分等。
-	Geometry模块：偏微分方程求解区域设定，包括空间N维、时间、多边形等。
-	Loss模块：网络求损失的模块，有监督无监督混合训练。
-	Operators模块：部分需要的算子在此定义。
-	Solver模块：包括求解器定义，偏微分方程问题定义等。

