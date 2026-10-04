# 第027单元 批次反向传播与损失归约

先修：[012 雅可比与矩阵求导](../012/README.md)、[024 逻辑回归与softmax分类](../024/README.md)、[026 标量计算图反向传播](../026/README.md)。中心问题：从一个样本推广到批次时，梯度和平均因子怎样变化？代码独立实现，不导入026。

## 学习材料

- [正文11页](lecture.pdf)：完整CE、仿射、激活矩阵梯度推导，shape、广播、微批归约、两层手算，8幅原创图与12题
- [独立实验指南3页](lab.pdf)，实验A至D；[完整详解4页](answers.pdf)
- [已执行Notebook](experiment.ipynb)：15个代码格、3张内嵌图，实际输出保留
- [NumPy实验脚本](experiment.py)、[12组独立测试](test_experiment.py)、[原创绘图程序](make_figures.py)
- 三份PDF的同名Markdown源；[来源说明](sources.md)、[来源核查记录](source-checks.json)、[验证记录](verification.json)
- data含5条合成数据与固定26参数配置；outputs含五份机器可读结果

## 运行

在本单元目录执行：

```bash
python experiment.py --output outputs
python test_experiment.py
```

主实验与测试只需Python和NumPy；测试另用标准库Decimal、Fraction和unittest。可参考环境声明：

```bash
conda env create -f environment.yml
conda activate dl-unit-027
jupyter lab
```

Notebook重启内核并顺序运行，写notebook_outputs后逐文件对照outputs。实际验证Python3.12.14、NumPy2.3.5、matplotlib3.10.8；Notebook为真实InProcessKernel、新Python进程顺序执行。未测试全新Anaconda、浏览器Jupyter、socket传输、跨平台或GPU。PyTorch仅核查官方文档，没有执行框架代码。没有网络下载、账号、付费API或外部数据依赖。

自定义数据/配置请用--data、--config、--output另存实验。固定图与Notebook入口核对默认两份输入和五份输出摘要，拒绝用固定解释套自定义结果。make_figures.py需要Noto Sans CJK字体；默认Linux字体路径写在脚本中，其他系统需按实际安装位置调整并重做视觉检查。PDF复建见[共享构建工具](../../shared/build-tools/README.md)。

## 核心结果

默认网络2→3→2→3，两个tanh隐藏层，共26参数、5样本。mean CE=1.148978865605775，sum=5.744894328028874，梯度范数≈0.2782642896。学习率0.1的单步后mean CE≈1.1413798716，只是固定点的一步下降。

全26坐标中心差分h=1e−5通过：最大绝对误差≈1.80e−11，最大带地板相对指标≈1.86e−9。独立逐样本标量平均最大差≈3.47e−17。2+3微批按2/5和3/5加权误差≈3.12e−17，简单平均两均值的梯度错误≈0.0330774。

两层手算ReLU模型mean CE=log2，第二层dW两行(−1,1)，db=(−1/2,1/2)；第一层dW为[[0,0],[−3/2,3/2]]、db=(−1,1)，dX全0。不同位置的零导数有不同抵消原因，不能据此认为整个网络没有参数梯度。

## 运行边界与失败保护

特征和参数绝对值≤10，批次1到128条，1到5个仿射层，宽度1到16，输出至少两类，激活/logit绝对值≤1e6。拒绝布尔、复数、NaN/无穷、错shape和越界/非整数标签。CSV和JSON及可检测扩展精度数组的非零转float64零会被拒绝；入口之前丢失的信息无法恢复。计算期舍入和下溢仍可能发生。

fd_h在1e−10到1e−2，learning_rate在1e−6到0.5；合法配置不保证差分、扰动或下降都成功，边界10的正扰动会越域。输入检查、计算和序列化先于最终写入，失败保持旧结果；有效运行覆盖五文件。每个文件替换独立，不保证操作系统级多文件原子性。

## 已完成的作者核查

72个70位Decimal前向模式参考网络、36组26参数与10输入坐标差分、120个Fraction仿射夹具、150组100位Decimal CE对照；手算、批次复制/置换/全部两段拆分、7种广播原shape、ReLU折点和错误梯度检出。23个数学拒绝、18种坏CSV/配置×新旧输出共36次新进程失败保护，以及计算失败隔离。

五结果在新进程、无关工作目录的-O执行和Notebook中逐字节一致。两个实验Python片段已执行；7个固定资料守卫与Notebook首格坏配置拒绝通过。18页最终PDF及8幅图均逐一目视检查。独立审阅与远端发布状态由课程总清单管理。
