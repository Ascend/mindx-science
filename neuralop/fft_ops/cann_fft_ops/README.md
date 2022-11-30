# FNO/PINO模型项目CANN侧自研FFT算子 （实虚部分离版本）

### 介绍
算子开发完全参照CANN侧AICPU算子开发流程，在原始的CANN代码仓中依下述说明添加或替换对应文件后，正常编译即可。

### 说明
1. 首先访问[此路径](https://gitee.com/ascend/canndev)，拉取CANN主仓代码。

2. **op_proto** 文件夹中的 *spectral_ops.h* 为CANN中频谱计算类算子的原型定义头文件，请用其替换原始的 *ops/built-in/op_proto/inc/spectral_ops.h* 文件。

3. **normalized** 文件夹存放本项目自研FFT算子的所有实现文件，请将其下所有文件直接添加至 *ops/built-in/aicpu/impl/kernels/normalized/* 目录下。

4. **aicpu_kernel** 文件夹中的 *aicpu_kernel.ini* 为算子信息库文件，请用其替换原始的 *ops/built-in/aicpu/op_info_cfg/aicpu_kernel/aicpu_kernel.ini* 文件。

5. 请用该目录下的 *CMakeLists.txt* 替换原始的 *ops/built-in/aicpu/impl/CMakeLists.txt* 文件。

6. 完成上述步骤后，请参照 [此文档](https://www.hiascend.com/document/detail/zh/aicpu_beta/aicpudevg_beta/atlasaicpu_10_0030.html) 中的指导完成编译部署。