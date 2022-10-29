# PINNworks

## NS 测试混合精度样例
1. clone本仓库到本地
2. 执行examples/NS/handleData.py ,随机采样生成数据集（data_input.npy,data_uv.npy）到data目录下。
3. （可选）执行examples/NS/showData.py,查看数据集的内容，数据类型。
4. 配置config.json中的设备，训练轮数等。
5. 执行train.py

### 一些关键函数
- 对于NS方程求导的配置在examples/NS/src/NS.py 中的governing_equation()
- 混合精度的开启在train.py 中的supervisedSolver，之前模型降精度的操作在train.py的66行


#### 介绍
pinn 模块化更新中


#### 软件架构
软件架构说明


#### 安装教程

1.  xxxx
2.  xxxx
3.  xxxx

#### 使用说明

1.  xxxx
2.  xxxx
3.  xxxx

