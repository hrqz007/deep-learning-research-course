# 线性模型训练闭环实验

## 第023单元 独立实验指南

目标是把一份固定拆分送入自己可解释的NumPy训练程序，证明梯度与更新正确，再比较合理常数基线。实验不使用GPU、框架自动微分、网络数据或付费API。先修009、011、014、020；正文给出所有推导。本文件不依赖提前查看详解，建议先完成每节的预测与手算。

## 一 建立环境并认识文件

核心脚本与测试依赖Python和NumPy。`environment.yml`另外列出Jupyter和可选绘图库，便于完整学习。在已安装Anaconda或Miniconda的终端中运行：

```bash
conda env create -f environment.yml
conda activate dl-unit-023
python experiment.py
python test_experiment.py
jupyter lab
```

应在本单元目录执行这些相对路径命令。核心实验运行时无需联网；第一次安装环境可能需要联网下载。已有其他Python环境也可使用，但先在其中确认NumPy可导入并核对版本。建议版本见环境文件，不是跨平台锁文件或“已经在你的电脑安装好”的声明。

作者实际使用Python 3.12.14、NumPy 2.3.5运行脚本和测试，并在新Python进程中通过真实IPython InProcessKernel按顺序执行Notebook。全新Anaconda安装、浏览器Jupyter界面和跨进程socket内核传输未实测。遇到这些层面的失败，应先检查环境，不要把它当作梯度公式的问题。

主要文件：`data/samples.csv`固定30行样本和拆分；`data/config.json`规定学习率0.1、300轮、零初值和梯度检查点；`experiment.py`包括核心函数与安全读写；`test_experiment.py`使用独立有理数参照；`experiment.ipynb`按学习顺序展示中间量。`outputs`保存实际脚本结果，Notebook另写`notebook_outputs`，后者仅是本地重跑副本。

## 二 先理解数组而不是抄公式

Python列表`[1,2,3]`不是本讲使用的数值向量对象。`np.array`将它转换为NumPy数组；`dtype=np.float64`选择双精度浮点。`.shape`是各轴长度组成的元组，`(3,)`表示一个轴长3，`(3,1)`表示两个轴，不能混淆。

```python
import numpy as np
X = np.array([[-1.0], [0.0], [1.0]])
y = np.array([-1.0, 1.0, 3.0])
A = np.column_stack((X, np.ones(len(y))))
theta = np.zeros(A.shape[1])
print(X.shape, y.shape, A.shape, theta.shape)
print(A @ theta)
```

应看到`(3,1)`、`(3,)`、`(3,2)`、`(2,)`，预测是三个零。`len(y)`返回3，`A.T`交换二维矩阵的两轴，`@`按行与列作内积，`*`则按相同位置相乘。不要用`X * theta`代替矩阵乘法；不要用`y.T`把一维向量“变成列”。

任务A1：不运行代码，先写出初始残差、半MSE与梯度。取步长1/2，写新参数、新预测及新损失。任务A2：用下面函数核对，并解释为什么轨迹有两行而只更新一次。

```python
from experiment import train
end, trace = train(X, y, learning_rate=0.5,
                   steps=1, initial=[0, 0])
for row in trace:
    print(row)
```

任务A3：将预测`np.array([-1.,1.,3.])`变成`[:,None]`，减去一维标签，查看残差形状与MSE。先预测结果，说明为何数值代码不报错却完全改变问题。

## 三 固定拆分并建立基线

打开`data/README.md`读取合成机制和字段含义。训练是12行、验证8行、测试10行，样本ID唯一，字段顺序固定。数据没有随机抽样步骤，重跑不需要重新生成拆分或设置随机种子。训练函数每次只得到训练数组。

```python
from pathlib import Path
from experiment import load_samples, split_arrays
rows = load_samples(Path('data/samples.csv'))
X_train, y_train = split_arrays(rows, 'train')
print(X_train.shape, y_train.shape, y_train.mean())
```

任务B1：证明最佳常数预测是训练均值，然后写出这个常数。任务B2：计算验证与测试MSE时应继续用这个值，还是各自重新取均值？给出信息边界理由。任务B3：手算或程序确认主例的$B=n^{-1}\sum_{ij}A_{ij}^2=31/6$。由充分下降界判断学习率0.1是否合适；说明超出这个保守界并不自动意味着发散。

本实验预先固定模型和预算，验证集不参与早停或参数搜索。看过测试输出后改变模型，只能称为一次新的探索，不应继续把原测试分数包装成首次独立评价。

## 四 核对梯度和数值最小二乘

```python
from experiment import gradient_check, least_squares
rows_check = gradient_check(X_train, y_train,
                            [0.4, -0.2, 0.7], h=1e-5)
for row in rows_check:
    print(row)
reference, info = least_squares(X_train, y_train)
print(reference, info['rank'], info['loss'])
```

任务C1：解释每个梯度检查字段。最大缩放误差应小于$10^{-8}$，但不要把它误写成真实任务预测误差。任务C2：把步长改为$10^{-100}$，解释为什么应被拒绝，而不是直接宣称梯度错误。任务C3：为什么参照使用`lstsq(A,y)`而不显式求逆？为什么应重新算残差，不能凭`residuals`为空宣布零损失？

执行默认脚本后打开`outputs/trajectory.csv`。任务C4：检查最后的梯度范数、目标与数值参照，确认301行与300次更新一致。允许浮点末期约$10^{-14}$量级的损失波动，不能强制所有位完全单调。主例在本环境中最终梯度范数约$6.58\times10^{-13}$，训练预测与参照最大差约$1.05\times10^{-12}$。

任务C5：从`samples.csv`和`predictions.csv`独立重算每种拆分的线性模型MSE与常数MSE，核对`summary.json`。训练循环的`loss`是半MSE，输出表中的`mse`是完整MSE。写一段结论，分别说明优化、基线提升及实验局限。

## 五 有控制地观察失败

所有尝试先在内存中做或另存目录，不要改默认图文数据。`train`支持一般小型二维特征和一维标签；脚本的CSV协议固定为两个特征。

```python
from experiment import train
for eta in [0.5, 2.0, 2.1]:
    end, trace = train([[0]], [1], eta, 22, [0, 0])
    print(eta, trace[0]['loss'], trace[-1]['loss'])
```

任务D1：这相当于只训练偏置。推导误差递推并预测三种行为。任务D2：令两列都为`[-1,0,1]`，标签为`[-1,1,3]`，从`[4,-2,0]`出发。解释为何两个权重差值始终为6；对照最小范数参照时要看哪些量？

任务D3：把配置复制为`my_config.json`，修改更新次数为30，运行下面命令。不要把30步的参数误称为默认300步结果。

```bash
python experiment.py --config my_config.json --output my_run
```

随后让固定图生成器读取`my_config.json`，它应该在写图前拒绝，因为现有图题属于默认配置。这是保护说明与数值一致性的检查，不代表30步实验非法。

任务D4：运行完整测试。它包含180个精确有理数梯度/多轮轨迹案例、100个独立精确最小二乘解、120个二次展开与差分案例，以及坏CSV、配置、计算、序列化和路径的隔离回归。检查无效输入时已有结果没有被改写；新输出目录也不应产生。测试使用精确有理数消元作小规模数学参照，未将浮点正规方程求逆作为稳定算法。

## 六 Notebook与报告验收

在本单元目录打开Notebook，重启内核后从第一格顺序执行。第一段计算先核对原始数据、配置与随附结果；不匹配会在Notebook输出前拒绝。不要删除这一步后仍使用固定解释。脚本默认路径相对脚本文件，支持从其他工作目录执行完整路径；Notebook的教学入口则明确要求单元目录。

本实验应交付：完整形状表与手算；每坐标梯度检查；训练轨迹与参照差异；三个固定拆分的同口径MSE表；一个关于广播或步长的失败例；一段优化成功不等于泛化证明的结论。重跑输出会覆盖指定目录中四个同名结果文件，另存请用`--output`。

全部输入、计算与序列化成功后才写结果，输出拒绝符号链接与输入冲突。文件逐个替换不是跨文件的崩溃事务；并发写、磁盘故障与操作系统级安装不属于这里已验证的范围。实验本身用CPU与原创合成数据，不需要账户或私人信息。
