# 024 实验：从logits到可验证的分类训练

用原创离线CPU合成点核对NumPy分类、梯度与极端logit；不做真实任务性能评估。

## 1 固定输入和形状

data/classification.csv包含9个train与6个test点。列是id、split、x1、x2、y，y为0、1、2之一。训练矩阵追加最后一列全1后shape为(9,3)，权重W为(3,3)，logits和概率均(9,3)，标签向量为(9,)。测试点不参与梯度、调参或迭代数选择。

data/config.json预定学习率0.25、400步、特征权重L2系数0.03，以及400步无正则二分类路径。正文0.3第一步手算是单独演示，不替代主配置。主脚本依赖Python与NumPy；测试还用SciPy作稳定参照。

```bash
python experiment.py --output outputs
python test_experiment.py
```

建议环境：

```bash
conda env create -f environment.yml
conda activate dl-unit-024
jupyter lab
```

作者实际使用Python3.12.14、NumPy2.3.5、SciPy1.17.0。环境文件是建议，不是测试过的全新Anaconda或跨平台锁文件；浏览器Jupyter界面和socket内核传输未测。Notebook实际在新Python进程、真实InProcessKernel中顺序执行。安装可能联网，实验本身不联网。

输出五文件：results.json保留配置、参数、手算梯度与极端例；training.csv每步记录总目标、纯CE、梯度范数和参数范数；binary_path.csv是一维可分例；predictions.csv逐条保留标签、类别与概率；environment.json记录运行版本。重跑会覆盖指定目录同名文件，自定义实验请另存。

## 2 先算一个二分类样本

取x=(2,−1)、w=(0,0)、b=0、y=1。写出z=0、p=1/2、损失log2，权重梯度(−1,1/2)、偏置梯度−1/2。学习率0.2更新得到w=(0.2,−0.1)、b=0.1，再计算新z=0.6。

```python
import math
from experiment import binary_loss_gradient, sigmoid
loss, grad = binary_loss_gradient([0.0], [1])
assert abs(loss - math.log(2)) < 1e-14
assert grad.tolist() == [-0.5]
new_loss = binary_loss_gradient([0.6], [1])[0]
assert new_loss < loss
print(sigmoid([0.6]), new_loss)
```

实验A：比较z=37,y=1时直接sigmoid(z)−1与函数梯度。前者可能为0，后者约−8.5330e−17。再用z=1000，解释为何稳定式也可能给0。不要把下溢的0解释成数学导数精确为0。

## 3 三类手算和稳定参照

z=(log2,0,0)、标签0时p=(1/2,1/4,1/4)，梯度(−1/2,1/4,1/4)。先手写再核验：

```python
import numpy as np
from experiment import multiclass_loss_gradient
loss, gradient, probability = multiclass_loss_gradient(
    [[np.log(2), 0.0, 0.0]], [0])
np.testing.assert_allclose(probability, [[0.5, 0.25, 0.25]])
np.testing.assert_allclose(gradient, [[-0.5, 0.25, 0.25]])
```

测试还对照SciPy的softmax、log_softmax、expit，以及100位Decimal自行构造的指数归一化。网页版本与本地库版本不同已记录，不能以读过最新文档代替本地实测。

实验B：对z=(1000,1001,−1000),y=2记录概率及稳定损失。第三类概率是浮点0，但损失仍约2001.31326169。再比较z=(8,0,0),y=0的正确CE与将概率送入同一logits接口的错误CE。不要在正式训练路径故意采用错误目标，只在隔离演示中计算它。

![实验图：极端logit与双softmax语义错误。](figures/04_logits_loss.png)

## 4 核对第一步和批次归约

读取train点，加常数1列，以全零W计算。检查初始损失log3，梯度为正文的有理数矩阵。正文单步示范用0.3且λ=0；完整训练用0.25和λ=0.03，二者配置要分别标记。

实验C：在同一批次、同一W且λ=0时，分别计算mean与sum损失/梯度。sum应为mean的9倍；用sum梯度和学习率0.25/9，应与mean梯度和0.25得到相同一步。接着保留两边相同非零λ，解释为什么只除学习率已不足以保持正则更新相同。

沿类别轴归一化后，每行概率应约和为1；解析logit梯度每行约和为0。不要误把“整个矩阵总和为1”当检查目标。接口拒绝(N,1)标签、浮点标签、bool标签、缺失类别和不匹配shape，避免广播把错误变成可运行程序。

## 5 梯度检查与训练结果

以中等尺度参数做中心差分，分别检查logits和W。每次只扰动一个坐标，保持同一损失定义、mean归约与L2正则。可以扫h从1e−1到1e−10，比较最大绝对误差。不要在极端饱和区只靠差分鉴别微小导数，也不要根据单次最小误差随意改实现。

训练400步后，训练CE约0.123658、测试CE约0.068982；训练9/9、测试6/6正确。总目标约0.238855，梯度范数约0.000309。保留所有逐步记录，不只截图准确率1。测试点是固定合成点，没有独立真实采样或总体性能保证。

实验D：证明一维二分类路径的损失 $L(w)=\frac12\log(1+e^{-w})+\frac12\log(1+e^{-2w})$ 对有限w严格下降，400步的w≈4.00124还不是有限零损失解。说明准确率从第一个正w开始已为1，为什么这不足以判断参数收敛。

## 6 可重跑范围与失败检查

脚本使用绝对值有界的float64数组：特征不超过100，权重不超过10000，logit不超过一百万。转换非零扩展精度或文本为float64时若变成0，会拒绝；调用者事先已经舍入丢失的信息无法恢复。函数不会替所有小概率强制截断到同一个epsilon，因为这会改变损失与梯度。

--config和--data可指定另存的配置/CSV，--output指定独立结果目录。字段、shape、有限性、类别、资源预算及所有计算/序列化完成后才写输出；输入或计算失败保留旧结果，不保证OS写入故障的多文件原子性。输出符号链接与覆盖输入位置被拒绝。

固定图和Notebook在任何展示/绘图库写入前核对原始数据、配置与五份默认结果摘要；不同输入/输出即拒绝，不用旧图解释自定义实验。可选make_figures.py需要matplotlib和Noto Sans CJK字体。PDF重建使用课程根shared/build_pdf_mathjax.py，分别传lecture.md、lab.md、answers.md，仍须逐页查看。Notebook请从本单元目录启动，重启并运行全部；它将主实验写到notebook_outputs，再和随附五文件逐字节比对。提交15题与A至D答案、五份产物及结论：必须区分概率形式和校准、训练目标和准确率、稳定浮点实现和精确实数、固定小例和真实任务。
