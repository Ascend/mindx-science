# PINN工具包

#### 介绍
pinn组件实现了求解正/逆偏微分方程的模块化通用工具包。其中通用模块包括：网络结构定义、数据采样生成、区域定义、损失函数生成及数据匹配、算子定义、求解器定义等。pinn组件还一并实现了求解含非线性项的一维含时Schrödinger方程、二维L型区域内的泊松方程、Navier-Stokes方程逆问题这三个偏微分方程，以作为参考样例。

#### 使用说明

在训练之前，首先请配置环境变量,将pinn_module添加至python环境变量。

    export PYTHONPATH=/{your path}/pinn_module

- NS方程
1. 从此[链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindx_science/pinns/cylinder_nektar_wake.mat) 下载数据cylinder_nektar_wake.mat，存放到examples/NS下。
2. 执行`python handle_data.py` ,随机采样生成数据集（data_input.npy,data_uv.npy）。
3. 配置config.json中的设备，训练轮数等。
4. 执行`python train.py`。

- Possion方程
1. 从此[链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindx_science/pinns/Poisson_Lshape_clean.npz) 下载数据Poisson_Lshape_clean.npz，存放到examples/Possion_Lshape下。
2. 配置config.json中的设备，训练轮数等。
3. 执行`python train.py`。

- Schrodinger方程
1. 从此[链接](http://mindx.sdk.obs.cn-snorth-4.myhuaweicloud.com/mindx_science/pinns/NLS.mat) 下载数据NLS.mat，存放到examples/Schrodinger下。
2. 配置config.json中的设备，训练轮数等。
3. 执行`python train.py`。


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

