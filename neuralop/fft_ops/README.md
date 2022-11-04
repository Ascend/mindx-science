# FNO/PINO模型项目CANN侧自研FFT算子安装流程

### 介绍
此文档中的 “安装” 意为将自定义的FFT算子功能扩充至 mindspore 框架，以供前端文件调用。

### 说明
1. 默认用户已安装 Ascend 版 mindspore，且知晓安装路径。

2. 第一步为获取内含自定义算子功能的动态库文件 (.so)，本目录下已经提供。若想手动编译，请克隆 Canndev 主仓代码后，参照 *fft_ops/cann_fft_ops* 中的 README 文件进行操作。编译完成后，将 *build/install/aicpu/libcpu_kernels_v1.0.1.so* 文件更名为 *libFFT_new.so* 即得。

3. 第二步，找到 mindspore 的安装路径。若使用 conda 安装，记所在的虚拟环境路径为根目录，则安装路径一般为 *~/lib/python3.x/site-packages/mindspore*。将上述 *libFFT_new.so* 拷贝至 *mindspore/lib* 即可。

4. 最后，能成功运行本项目的自研模型即代表安装成功。