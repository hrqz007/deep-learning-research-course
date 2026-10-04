# 027 实验：批次反传的三路核验

本实验把“矩阵公式正确”变成可复查的证据：手算两层网络；矩阵反传对照逐样本标量求和；对全部26个参数进行有限差分；最后故意破坏归约，确认测试能发现错误。先修012、024、026，配合本讲正文与独立答案阅读。

## 准备：目录、环境与目标

打开本单元目录。`data/batch.csv`包含5条原创合成记录，列为id、x1、x2、y；它是导数夹具，没有训练/测试划分。`data/config.json`固定2→3→2→3网络、tanh、差分基准步长$10^{-5}$及一次更新的学习率0.1。输入、参数和标签在核验过程中保持不变。

主脚本仅需Python和NumPy，测试还用标准库Decimal、Fraction和unittest。Notebook需要Jupyter内核与IPython，绘图需要matplotlib。`environment.yml`给出环境声明；可用以下命令创建自己的环境，但本次没有在全新Anaconda中执行安装：

```bash
conda env create -f environment.yml
conda activate dl-unit-027
python experiment.py --output outputs
python test_experiment.py
jupyter lab
```

打开`experiment.ipynb`，重启内核后从第一格顺序执行。Notebook先核对默认数据与已发布默认结果摘要，再把新结果写入`notebook_outputs`并逐文件比较。自定义实验使用单独输出目录；固定Notebook与固定图的文字不能自动适配修改后的数据。

实际验证环境为Python3.12.14、NumPy2.3.5，Notebook通过真实InProcessKernel在新Python进程顺序执行并保存输出。不声称测试过浏览器Jupyter、socket内核、全新Anaconda、GPU或PyTorch。相同环境结果可逐字节核对；不同平台应同时看数值容差。

## 实验A 手算后再运行

用正文第五节的2→2→2 ReLU网络，先写出$H^{(1)},Z^{(2)},P,D^{(2)}$，再求两个$W$、两个$b$和输入$X$的梯度。明确目标为两条样本CE的mean，检查每个shape。

预期：平均损失为$\log2$；第二层权重梯度两行都是$(−1,1)$；第一层权重梯度第一行全0，第二行为$(−3/2,3/2)$。不要据此认为第一层“没有学到信号”，继续检查偏置梯度并解释抵消来源。将目标换为sum，所有梯度应乘2。

测试`test_hand_network`包含这组完整解析期望，也逐参数做数值核验。提交你自己的展开过程；只贴测试通过不算完成手算。

## 实验B 默认网络的完整梯度审计

运行主脚本，打开五份输出：

| 文件 | 检查什么 |
|---|---|
| summary.json | 目标、梯度范数、误差摘要、微批重建与单步结果 |
| gradients.csv | 全部26个参数的解析/数值导数、绝对/相对误差与判据 |
| sample_gradients.csv | 5条样本各26个未平均导数，共130条记录 |
| fd_sweep.csv | 9种步长下最差误差与不通过坐标数 |
| tensors.json | 所有层的H、Z、D、概率、输入梯度和参数梯度 |

确认26坐标没有漏掉偏置。对逐样本CSV中同一参数的5个数求平均，与`gradients.csv`解析列比较。不要把5×26个数整体平均成一个标量。默认应得到mean CE约1.1489788656、梯度范数约0.2782642896、逐样本核验最大差约$3.47\times10^{-17}$。

可在单元目录运行下面片段，打印每层关键shape：

```python
from experiment import load_inputs, loss_and_grad
_, X, y, params, cfg = load_inputs()
r = loss_and_grad(X, y, params, cfg['activation'])
for j, p in enumerate(params):
    print(j+1, r['H'][j].shape, r['Z'][j].shape,
          r['delta'][j].shape,
          r['grads'][j]['W'].shape,
          r['grads'][j]['b'].shape)
```

逐参数中心差分使用$h_r=h\max(1,|\theta_r|)$。提交最差坐标的名字及两种导数，而不只是最大误差。默认$h=10^{-5}$时最大绝对误差约$1.80\times10^{-11}$，26坐标均通过。扫描表中较小步长不保证更好；解释误差回升所体现的数值相减问题。

## 实验C 两种微批组合与三种故障

不改参数，把前2条与后3条分别求mean梯度。计算$\frac25\bar g_A+\frac35\bar g_B$和$\frac12(\bar g_A+\bar g_B)$，与完整批次mean比较。默认正确版本最大差约$3.12\times10^{-17}$，错误版本约0.0330774。

然后做三次独立干预，每次都从同一份原始参数开始：

1. 忘记输出端的`/n`。检验梯度是否整体变为正确值5倍。
2. 输出端已除$n$，偏置梯度又把sum改成mean。检查每层偏置是否缩小5倍；不要真的把错误函数覆盖到正式发布脚本。
3. 对3样本3类的零logits和标签$(0,0,1)$，分别沿轴0与轴1求和。两者shape相同，数值却不同，说明只检查shape不够。

以下片段可直接复现第三种故障，不修改任何生产文件：

```python
import numpy as np
from experiment import cross_entropy
_, delta, _, _ = cross_entropy(np.zeros((3, 3)), [0, 0, 1])
print('correct db:', delta.sum(axis=0))
print('wrong axis:', delta.sum(axis=1))
```

附带测试还将整批复制3次、打乱行序，并检验每个可能的两段拆分。提交这些不变量为什么成立。说明如果每个微批之后立刻更新参数，为什么再累加已不能等价。

## 实验D 失败输入与不可靠差分

复制数据或配置到一个临时目录，针对至少三个错误分别运行：CSV标签3越界、标签变成`0.0`、配置学习率为布尔值true、参数错shape或CSV特征文本`1e-400`。输出使用临时目录，保留一套旧结果的摘要作为对照。

主脚本应以非零退出码报告问题；新输出目录不应在输入失败后产生，旧五文件不应改变。附带测试用18种坏CSV/配置分别核对新旧输出，共36个新进程运行，另检查计算期失败。不要拿操作系统断电或磁盘故障测试结果外推成多文件原子性保证。

另单独计算ReLU在0的中心差分，得到0.5；程序反传约定为0。解释这次差异为何不能用来证明一般反传写错。随后查看tanh模型的9步长误差曲线：它是一条核验诊断线，不是训练损失曲线。

## 提交清单与结论范围

提交手算过程、五份默认输出、执行后的Notebook、最差差分坐标、微批两种组合和三种故障的定位解释。保留自定义实验配置与数据，用独立目录存结果。正式默认文件不作为草稿覆盖。

测试通过只支持这组函数和已检验点附近的正确性；单步损失下降不等于收敛，批次梯度一致不等于数据有效。来源、数值与运行边界见正文、source-checks.json、verification.json和README。完整练习及A至D答案另见answers.pdf。
