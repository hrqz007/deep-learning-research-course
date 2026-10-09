# DL077 实验：模型为何依赖一个无因果效应的字段

对应原大纲 T5。你将训练一个2→12→1小网络，手算与检验梯度、积分梯度，随机化参数，并在已知生成机制里比较观察关联与干预效应。全部数据本地生成，没有外部模型或数据服务。

## 1 环境与首轮运行

在本单元目录执行以下命令。首次安装需要软件包下载；安装后实验离线。Python3.12是已验证版本，建议新建独立环境，避免与其他课程依赖混用。

```bash
# 安装NumPy和Notebook工具；不下载训练数据。
python -m pip install -r requirements.txt
# 验证解析梯度、积分完整性与因果机制；预期10项通过。
python -m unittest -v test_experiment.py
# 确认优化模式下检查仍生效；预期10项通过。
python -O -m unittest -v test_experiment.py
# 真实训练1800步并另存数据、权重、预测和指标。
python experiment.py --output my_run
```

`my_run/results.json` 应含预测、gradient、ig_zero、ig_alternative、randomization、perturbation、causal 等字段。测试RMSE约0.089，真实do(X)每单位效应2，do(Z)效应0。模型对Z加1的平均响应约0.993。不要把这些数写成“因果网络准确率”；它们检验不同问题。

数据字典见 [data_dictionary.md](data_dictionary.md)。train_x是600×2，列0为X，列1为Z；train_y是600项。外生表四列依次U、εx、εy、εz。测试有400条且种子独立。训练网络不读U；U仅在机制验证和调整回归中使用。

### 一眼看懂信息边界

Z=Y+小噪声，是Y发生后的测量，因此预测任务刻意容易。本实验要检验“预测依赖不推出因果”，不是制造一个可部署的提前预测系统。如果把此数据表当成未来Y预测项目，应识别出Z在决策时不可用。训练脚本能正确运行与业务信息可用性是不同审查层次。

打开 `experiment.py`，先阅读 `generate` 的三条方程，再看 `predict`、`gradient`、`ig`。不要先看条形图猜故事。纸上标出W为2×12、v为12项，确认两个输入沿12条路径到达同一个标量输出。

<div style="break-before:page"></div>

## 2 手算、数值与反驳实验

**练习1：链式法则。** 单隐藏单元 f=3tanh(2x₁−x₂)+1。在 x=(0,0) 时计算两个输入偏导数，并用中央差分验证。解释输出偏置1为何不影响梯度。

**练习2：积分梯度。** 对线性函数 f=2x₁+3x₂，目标点(1,2)，分别从(0,0)和(1,0)出发计算IG。核对每次归因和等于哪个输出差。为何不能把两次归因变化直接解释成模型变了？

**练习3：单位变换。** 假设第一个输入以米计，现改为厘米s=100x₁，定义同一预测函数 g(s,x₂)=f(s/100,x₂)。推导g对s的梯度，比较物理上相同改动的输出变化。只看原始梯度大小排序会发生什么？

```python
# 读取可信本地数字权重，不启用Python对象反序列化。
import numpy as np
# 引入已注释的预测、导数、积分函数。
from experiment import predict, gradient, ig
# weight文件中p0至p3分别是W、b、v、c。
w = np.load('my_run/weights.npz', allow_pickle=False)
p = [w[f'p{k}'] for k in range(4)]
# 设置固定查询点与明确的零基线。
x, b = np.array([0.8, 4.0]), np.zeros(2)
# 打印两个字段的路径归因及其完整性误差。
a = ig(x, b, p, steps=512)
print(a, float(a.sum() - (predict(x[None], p)[0] - predict(b[None], p)[0])))
```

**练习4：参数随机化。** 保持查询点、基线、纵轴不变，分别展示训练网络、随机输出层、全随机网络的带符号IG。同时看80个探测点的相对变化与余弦。解释“随机化后变化很大”能支持什么，不能支持什么。

**练习5：输入扰动。** 使用0.001、0.01、0.1、1四个增量，比较Z扰动的实际输出变化与梯度乘增量。若本次大步仍近似一致，如实写“这条路径曲率较小”，不要假设所有大步必然明显失败。

<div style="break-before:page"></div>

## 3 干预设计与最终报告

**练习6：手算混杂。** U方差1，εx方差0.16，X=U+εx，Y=2X+3U+εy。推导Y对X一元观察回归的总体斜率，并与真正do(X)效应比较。解释为何多加样本只能更精确估计观察斜率，不能把它自动变成2。

**练习7：一致的结构干预。** 固定某行U、εy、εz，分别设X=−0.5和0.5；按顺序重算Y、Z。再只把Z加1而保持Y不变。写出两种干预不同的方程替换位置。指出只动模型输入X而保留旧Z为何不是同一个完整生成干预。

**练习8：换一个问题。** 删除Z重训，只用X预测Y；不要直接改原脚本覆盖固定证据，应另存探索副本。预期模型会倾向拟合观察关系还是因果系数？你的结论需要哪些调整假设或新的干预数据？允许用线性回归作为简单对照。

```bash
# 重建Notebook（清除旧输出），保存修改前请先备份自己的笔记。
python create_notebook.py
# 顺序执行所有单元，Notebook另存notebook_outputs并完整训练。
python execute_notebook.py experiment.ipynb
# 根据固定outputs重建六张图，不访问网络。
python make_figures.py
# 有文档构建依赖后，分别生成三份PDF。
python build_pdf.py lecture.md
python build_pdf.py lab.md
python build_pdf.py answers.md
```

文档构建依赖与实验依赖不同，见 [build-tools/README.md](build-tools/README.md)。进程内内核执行可验证顺序与变量依赖，未声称验证浏览器交互。无论通过哪种执行器，都要先清空旧输出重跑，不能保留失败之前的漂亮结果作为“已验证”。

最终交付一页解释报告与一张机制图：明确输出、样本、基线、积分步数、随机化范围、相似度、扰动尺度、生成方程、do操作和识别假设。把“观察到什么”“模型依赖什么”“干预改变什么”分成三句，每句紧跟支持证据。

评价：25%梯度与IG推导、25%参数/输入检查、30%干预与混杂推理、20%可复算报告及信息边界。发现“Z是事后变量”应计为重要理解，而不是为了低RMSE回避它。参考答案完整解释每道题，并给出不能由本实验推出的结论。
