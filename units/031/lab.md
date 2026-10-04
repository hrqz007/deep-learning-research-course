# 031 实验：把训练失败缩成证据

本实验交付一份可定位故障报告，而不是一张“训练终于下降”的图。你将实际执行：逐坐标有限差分、框架gradcheck、参数梯度与更新对照、固定单批次拟合、随机标签拟合、矛盾标签下界、输入扰动。所有数据均人工合成，CPU即可运行。

## 1 准备与边界

从本目录打开终端，按`README.md`创建或选择Python环境。所需核心版本为Python3.12、NumPy2.3.5、PyTorch2.7.1 CPU；Notebook还需要Jupyter与ipykernel。不要在脚本里加入下载安装。可先执行：

```bash
python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
python experiment.py --output outputs
python -m unittest -v test_experiment.py
```

脚本以自身位置寻找输入，因此可从其他工作目录启动。`--data-dir`只用于指定同一套固定输入的存放位置；字节发生变化会被拒绝，包括只改空白。它不是自由配置调参接口。自由探索可导入`train_one_batch`等函数，使用新的数组和输出位置，不改教材参考结果。

Notebook先核验已随教材提供的输入与7份归档结果，再独立重算到`notebook_outputs/`并逐文件比较。图像展示的也是经过摘要核验的固定参考。选择“重启内核并运行全部”；逐格挑着运行不构成复现。

本环境实际验证过新的Python进程与新的IPython InProcessKernel顺序执行；没有验证浏览器Jupyter界面、网络socket内核传输、GPU或全新Anaconda安装。这些限制不影响本讲CPU计算证据，但不能省略。

## 2 任务A：先验证数据和一个手算例

打开`data/batch.csv`，核对8个ID和两列输入。逐行验证目标等于$0.7x_1-0.4x_2+0.2$；第二组标签是固定随机回归标签，不是类别编号。用纸笔计算线性例$x=(-1,1),y=(-1,1),w=0.5,b=0.25$：两个预测、两个残差、half-MSE、两个梯度和学习率0.1的一步更新。

再运行以下完整片段，故意制造标签shape错误：

```python
import torch
p = torch.tensor([[1.], [2.]], dtype=torch.float64)
y = torch.tensor([[1.], [2.]], dtype=torch.float64)
print((p-y).shape, 0.5*(p-y).square().mean())
print((p-y[:, 0]).shape, 0.5*(p-y[:, 0]).square().mean())
```

应看到`(2,1)`对应0，`(2,2)`对应0.25。解释为什么第二行没有报错。提交一条数据契约：“预测与标签都必须是(n,1)，不得依赖广播纠正语义。”区分这一语义错误与NaN、无穷大等数值错误。

## 3 任务B：完整前后向，再逐坐标检查13个参数

先按正文3.1–3.5节，从相同的4条样本与13参数走完一次前向、损失、全部局部导数、样本路径贡献、总梯度、SGD更新与新前向。对照`outputs/full-trace.json`，逐项检查A、H、预测、残差、R、tanh局部导数、dH、dA、4乘13贡献、13梯度以及更新后结果。不能只提交一个最大误差数；至少写出W1[1,1]的四样本贡献和b1[1]为何与它不同。Notebook会展示所有坐标。

`experiment.py`中的`scalar_reference`用Python循环独立实现前向与链式法则。`autograd_values`使用同一个数学网络的PyTorch实现。中央差分只调用前者的前向值，避免用反向结果自我验证。

```python
import numpy as np
from experiment import load_inputs, autograd_values, central_difference
X, y, _, cfg = load_inputs()
loss, gradient = autograd_values(X[:4], y[:4], cfg["theta"])
numeric = central_difference(X[:4], y[:4], cfg["theta"], 1e-6)
print(loss, np.max(np.abs(gradient-numeric)))
```

查看`outputs/finite-difference.csv`的78行：13坐标乘6个步长。用`abs_error <= 1e-8 + 1e-5*max(abs(a),abs(b))`重建通过标记。记录哪个步长区间较好；不能把所有步长都必须通过作为验收条件。

![实验图1 差分误差随步长变化。中间区间提供数值一致性，极小步长相减误差增大不是自动求导已经坏掉的证据。](figures/03_fd_steps.png)

查看`outputs/faults.json`：正常gradcheck通过，隐藏层分离不通过，标签反转仍通过。提交失败坐标到参数组的映射。再解释ReLU在0的0与0.5为何不能用同一通过条件判框架对错；解释单方向投影相同为何还可能梯度向量不同。

## 4 任务C：诊断梯度有了但参数不动

对前4条样本运行六个相同初始状态的一步探针。每条记录包括参数名字、是否在优化器内、梯度是否为`None`、梯度范数与参数更新范数。注意所有探针都是无动量SGD，学习率0.03；`zero_lr`故意改成0。

```python
from experiment import load_inputs, one_step_probe
X, y, _, _ = load_inputs()
for mode in ("healthy", "no_step", "zero_lr", "omit_W1",
             "detach_hidden", "sum_reduction"):
    report = one_step_probe(X[:4], y[:4], mode)
    print(mode, report["parameters"][0])
```

完成一张自己的诊断表：哪些故障保持所有梯度却停止所有更新？哪个只停止W1？哪个令W1、b1都没有梯度？哪个恰好放大4倍？选择两个症状相似的故障，给出区分它们所需的额外观察。不要仅凭更新为0就断言梯度断了。

## 5 任务D：单批次、随机标签与不可拟合例

查看`outputs/training.csv`，两组标签各有1601行：行0是更新前初始状态，行1600是完成1600次更新后的诊断，不执行第1601次更新。末行`update_norm=0`是这个记录约定，不是优化器故障。

1. 两次训练都是49参数、相同初始化、Adam学习率0.03。核对初始和末尾loss，与`summary.json`一致。
2. 在`fitted-values.csv`逐样本检查预测与各自标签，不只看聚合指标。
3. 将两条相同输入配上-1和1，先证明最小half-MSE为0.5，再核对实际300步结果。
4. 写出一句恰当结论：“本固定结构和更新条件能拟合这8条约束。”另写一句不能支持的结论，并说明缺什么证据。

![实验图2 两次小批次拟合。随机标签拟合用于诊断容量和链路，不能用于选择真实任务中的模型。](figures/06_batch_fits.png)

如要自己重做，可调用`train_one_batch(X, y, steps=1600, lr=0.03, seed=31)`。函数返回训练好的模型及全部轨迹；不要把返回的轨迹误当独立测试成绩。修改种子或预算后，不预先承诺一定达到相同阈值。

## 6 任务E：冻结模型做输入干预

完整脚本已经记录以下干预。你应说明每次改了什么、保持了什么：输入标签联合反转；只反转输入；全部输入置零；每个输入的第一坐标增加0.1、0.01、0.001。都不重新训练模型。

在`perturbations.json`中核对联合置换不改loss；只改输入顺序时loss约0.975；置零后约0.243756。后三个小扰动的平均变化率约0.58414、0.58387、0.58383，而构造规则的变化率为0.7。解释为什么训练点几乎零误差与这个差别并不矛盾。

再检查`loss_against_shifted_rule`。扰动越小，loss越小，但平均变化率并未趋向0.7。用“偏差乘扰动尺度”的关系说明原因。不可把这一小型干预报告写成鲁棒性、因果识别或泛化性能报告。

## 7 最终提交：一份可定位的故障报告

选`omit_W1`或`detach_hidden`，提交不超过两页的独立报告，含以下内容：

- 任务、数据ID、初始状态、shape、dtype、版本与重跑命令
- 一句明确预期，例如“W1应参与同一次SGD更新”
- 实际loss、梯度存在性、参数梯度范数、参数变化、优化器成员身份
- 最早偏离预期的层，排除了什么，仍未排除什么
- 只改一处的修复方案与修复后的同样证据
- 一条自动化回归断言，以及此报告不能证明的统计结论

推荐回归用`unittest`的`assertAlmostEqual`等方法；不要把生产输入验证写成Python的`assert`，因为`python -O`会移除普通assert语句。本讲脚本使用显式异常检查。

验收重点：数学定义与代码一致；能定位而不只是观察；有真实对照与失败反例；诚实限定结论。完整答案在`answers.pdf`。先自己提交证据，再看参考答案。
