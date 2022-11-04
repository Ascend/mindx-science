# FNO & PINO MindSpore ver.

## 模块介绍

主要介绍 nerualop 内部结构

- _fno_ex 

  和 FNO 相关训练和测试入口脚本, 需要从命令行接收 `config_path` 参数来读取对应 yaml 配置文件来确定模型名称、数据集和网络超参数

  - train_fno 
  
    训练 FNO

  - eval_fno

    测试 FNO

- _pino_ex 

  和 pino 相关训练和测试入口脚本, 需要从命令行接收 `config_path` 参数来读取对应 yaml 配置文件来确定模型名称、数据集和网络超参数

  - train_pino 
  
    训练 PINO

  - eval_pino

    测试 PINO

- architecture

  网络原型

  - basic_block

    包含网络基础组件, 主要为一到三维的频域卷积层

  - fft_ops_1

    自研 FFT 算子的前端文件

  - fourier1d

    FNO1d 网络模型

  - fourier2d

    FNO2d, FNO2dtime, PINO2d 网络模型

  - fourier3d

    FNO3d, PINO3d 网络模型

- configs

  配置文件, 配置模型名称、数据集和网络超参数

- data

  数据集预处理

  - datasets

    包括所有模型所需数据集及其预处理. 大部分问题支持下采样, 即抽取稀疏化的网格. 部分问题提供更复杂的预处理, 如 PINO3d, 会对原数据集做截取和扩充

- loss

  损失函数

  - losses

    包括所有模型的损失函数, 主要包括: LPLoss(计算通用 $L^p$ 范数)、 FNO2dtime 的增量运算损失函数、 PINO2d 和 PINO3d 的基于数据或方程的损失函数

- utils

  辅助函数

  - utils

    主要包括自定义学习率调节器、 自定义回调函数和自定义性能分析上下文管理器

## 数据集描述

- FNO1d Burgers 方程

  [burgers_data_R10.mat](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing)

  包含字段
    - 输入 `a` 2048\*8192
    - 输出 `u` 2048\*8192
  
  其中 2048 为样本数, 8192 为网格数

  使用前 1000 个样本训练, 后 100 个样本测试

- FNO2d Darcy Flow 问题

  [piececonst_r241_N1024_smooth1.mat](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing)

  包含字段
    - 输入 `coeff` 1024\*241\*241
    - 输出 `sol` 1024\*241\*241
  
  其中 1024 为样本数, 241 为网格数(x 和 y 方向)

  [piececonst_r241_N1024_smooth2.mat](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing)

  包含字段
    - 输入 `coeff` 1024\*241\*241
    - 输出 `sol` 1024\*241\*241
  
  其中 1024 为样本数, 241 为网格数(x 和 y 方向)

  使用第一个数据集的前 1000 个样本训练, 第二个数据集的前 100 个样本测试

- FNO2dtime/FNO3d Navier-Stokes 方程

  [NavierStokes_V1e-5_N1200_T20.mat](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing)

  包含字段
    - `u` 1200\*64\*64\*20

  其中 1200 为样本数, 64 为网格数(x 和 y 方向), 20 为时间网格数

  使用前 1000 个样本训练, 后 100 个样本测试

- PINO2d Darcy Flow 问题

  [piececonst_r421_N1024_smooth1.mat](https://hkzdata.s3.us-west-2.amazonaws.com/PINO/piececonst_r421_N1024_smooth1.mat)
  
  包含字段
    - 输入 `coeff` 1024\*421\*421
    - 输出 `sol` 1024\*421\*421
  
  其中 1024 为样本数, 421 为网格数(x 和 y 方向)

  [piececonst_r421_N1024_smooth2.mat](https://hkzdata.s3.us-west-2.amazonaws.com/PINO/piececonst_r421_N1024_smooth2.mat)
 
  包含字段
    - 输入 `coeff` 1024\*421\*421
    - 输出 `sol` 1024\*421\*421
  
  其中 1024 为样本数, 421 为网格数(x 和 y 方向).

  使用第一个数据集的前 1000 个样本训练, 第二个数据集的前 100 个样本测试


- PINO3d Navier-Stokes 方程

  [NS_fft_Re500_T4000.npy](https://hkzdata.s3.us-west-2.amazonaws.com/PINO/NS_fft_Re500_T4000.npy)

  包含数组: 4000\*65\*64\*64 
  
  其中 4000 为样本数, 64 为网格数(x 和 y 方向),  65 为时间网格数

  [NS_fine_Re500_T128_part2.npy](https://hkzdata.s3.us-west-2.amazonaws.com/PINO/NS_fine_Re500_T128_part2.npy)

  包含数组: 100\*129\*128\*128
  
  其中 100 为样本数, 128 为网格数(x 和 y 方向),  129 为时间网格数

  使用第一个数据集的前 100 个样本训练(内部会将每个样本截取并扩张成 4 个), 第二个数据集的所有 100 个样本测试

## 如何使用

准备主仓库代码和 PyTorch 版代码, 其中PyTorch 版代码基于论文作者给出的代码稍作修改而成

```bash
git clone https://gitee.com/burning489/mindx-science.git
cd mindx-science/neuralop
git clone https://gitee.com/burning489/neuralop_torch.git torch
```

1. MindSpore 版

  - 训练

  ```bash 
  python _fno_ex/train_fno.py --config_path configs/fno1d_pretrain.yaml
  python _fno_ex/train_fno.py --config_path configs/fno2d_pretrain.yaml
  python _fno_ex/train_fno.py --config_path configs/fno2dtime_pretrain.yaml
  python _fno_ex/train_fno.py --config_path configs/fno3d_pretrain.yaml
  python _pino_ex/train_pino.py --config_path configs/pino2d_pretrain.yaml
  python _pino_ex/train_pino.py --config_path configs/pino3d_pretrain.yaml
  ```

  - 测试

  ```bash 
  python _fno_ex/eval_fno.py --config_path configs/fno1d_pretrain.yaml
  python _fno_ex/eval_fno.py --config_path configs/fno2d_pretrain.yaml
  python _fno_ex/eval_fno.py --config_path configs/fno2dtime_pretrain.yaml
  python _fno_ex/eval_fno.py --config_path configs/fno3d_pretrain.yaml
  python _pino_ex/eval_pino.py --config_path configs/pino2d_pretrain.yaml
  python _pino_ex/eval_pino.py --config_path configs/pino3d_eval.yaml
  ```

  - 性能分析

  ```bash 
  python _fno_ex/train_fno.py --config_path configs/fno1d_prof.yaml
  python _fno_ex/train_fno.py --config_path configs/fno2d_prof.yaml
  python _fno_ex/train_fno.py --config_path configs/fno2dtime_prof.yaml
  python _fno_ex/train_fno.py --config_path configs/fno3d_prof.yaml
  python torch/pino2d.py --config_path configs/pino2d_prof.yaml
  python torch/pino3d.py --pretrain_path configs/pino3d_prof.yaml --eval_path configs/pino3d_eval.yaml
  ```

2. PyTorch 版

  - 训练和测试

  ```bash 
  python torch/fno1d.py --config_path configs/fno1d_pretrain.yaml
  python torch/fno2d.py --config_path configs/fno2d_pretrain.yaml
  python torch/fno2dtime.py --config_path configs/fno2dtime_pretrain.yaml
  python torch/fno3d.py --config_path configs/fno3d_pretrain.yaml
  python torch/pino2d.py --config_path configs/pino2d_pretrain.yaml
  python torch/pino3d.py --pretrain_path configs/pino3d_pretrain.yaml --eval_path configs/pino3d_eval.yaml
  ```

  - 性能分析

  ```bash 
  python torch/fno1d.py --config_path configs/fno1d_prof.yaml
  python torch/fno2d.py --config_path configs/fno2d_prof.yaml
  python torch/fno2dtime.py --config_path configs/fno2dtime_prof.yaml
  python torch/fno3d.py --config_path configs/fno3d_prof.yaml
  python _pino_ex/train_pino.py --config_path configs/pino2d_prof.yaml
  python _pino_ex/train_pino.py --config_path configs/pino3d_prof.yaml
  ```

## 如何从零配置, 以 PINO3d 为例

在本项目中, 该模型用于求解 N-S 方程, 本节将介绍如何配置一个 pino3d.yaml

1. 数据集

  以下部分需配置于 `data` 字段内

  在此我们选取作者开源提供的数据集, 数据集规格详见第一节介绍

  我们全程会使用三个数据集: 训练集、测试集和高斯随机场生成的随机初始条件数据集

  训练集需提供 
  - `train_path` 指向数据集所在路径
  - `Re` 表示该 N-S 方程的雷诺数
  - `offset` 表示数据集偏移量, 即从第几个样本开始读取, 默认为 0 即可
  - `ntrain` 指定训练样本数
  - `time_interval` 提供求解的时间长度
  - `nx` 提供空间原始网格数
  - `nt` 提供时间原始网格数
  - `sub` 提供空间下采样率
  - `sub_t` 提供时间下采样率
  - `nx_data` 提供下采样后的空间网格
  - `nt_data` 提供下采样后的时间网格
  - `batch_size` 指定数据集每批的样本数
  - `shuffle` 指定是否打乱数据集.

  测试集配置与训练集类, 仅有 `test_path` 和 `ntest` 命名差异.

  高斯随机场生成的随机初始条件数据集需提供
  - `nx_eqn` 指定人造随机数据集的空间网格数
  - `nt_eqn` 指定人造随机数据集的时间网格数

2. 网络

  以下部分需配置于 `model` 字段内

  需提供
  - `name` 指定模型名, 在 PINO 部分可选值为 pino2d 和 pino3d, 此处我们选择 pino3d
  - `layers` 提供列表指定网络全连接层的深度和宽度
  - `modes1`、`modes2` 和`modes3` 分别指定各个维度的 FFT 截断频率
  - `fc_dim` 指定倒数第二层全连接层的宽度
  - `activation` 指定激活函数, 可选 tanh、 relu 或 gelu.

3. 训练

  以下部分需配置于 `train` 字段内

  提供
  - `mix_prec` 是否开启混合精度, 根据实测效果, 我们建议设置为 true
  - `epoch` 指定训练迭代次数
  - `base_lr` 指定基础学习率
  - `weight_decay` 指定 Adam 优化器 weigth_decay 参数
  - `ic_weight` `f_weight` `xy_weight` 指定最终损失函数中基于初始条件、方程本身和数据的损失函数的权重
  - `num_data_iter` `num_eqn_iter` 在每次大迭代中, 基于数据和基于人造随机初始条件的训练次数
  - `load_path` 可选, 可以读取已有 ckpt 继续训练
  - `save_per_epoch` 可选, 用于回调函数, 指定每过多少迭代次数保存一次网络参数
  - `save_path` 可选, 网络参数保存路径
  - `profile` 可选, 为 false 时仅有训练计时, true 时会开启性能分析
  - `profiler_path` 指定当 `profile` 为 true 时, 性能分析报告保存的路径
  - `decay_per_epoch` 对于除 PINO3d 外的其他模型, 支持学习率调节器, 可每 `decay_per_epoch` 次迭代后衰减学习率
  - `scheduler_gamma` 学习率衰减比例

4. 测试

  需配置于 `test` 字段内, 需提供 `load_path` 指定读取的 ckpt 路径