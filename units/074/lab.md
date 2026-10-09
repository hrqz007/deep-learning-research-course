# DL074 实验：可以检查的泛化保证

对应原大纲T2。目标是分开“总体风险”“经验风险”“理论半径”和“有限重复频率”，并检查数据依赖模型选择。预期90分钟，CPU可离线执行，数据由本单元公式生成，不需要真实用户或外部下载。

## 1 固定实验问题与输入

先阅读experiment_plan.json与data/README.md。总体是21个等概率格点，标签条件概率为0.15或0.85。23个预定阈值含一对重复函数，使用M=23作保守计数。不能观察训练标签以后缩小候选清单，再把新大小代回原界。

```bash
# 首次安装软件依赖可能联网，实际数据和训练不联网。
python -m pip install -r requirements.txt
# 写出精确总体数组和100条固定样例CSV。
python generate_data.py
# 运行12项定义、手算、边界与确定性测试。
python -m unittest -v test_experiment.py
# 在Python优化模式重复12项测试。
python -O -m unittest -v test_experiment.py
# 重新进行4组各400次真实抽样，保留到自选目录。
python experiment.py --output my_run
```

预期生成results.json和trials.npz；每种样本量都报告400次，不只报告平均数。sample.csv是教学用样例，不是1600份实验样本的拼接；完整实验可由固定种子规则重新生成，各次风险结果保存在NPZ中。

## 2 先做三个手算

任务A：某规则在三个等概率输入上的预测为[0,1,1]，真实标签为1的条件概率为[0.2,0.7,0.4]。不抽任何测试集，直接计算总体风险。写出每个输入的条件错误概率，再平均。

任务B：M=23，n=100，delta=0.05，计算epsilon。若观察到训练风险0.12，给出[0,1]内的总体风险保证区间。说明这不是对已观察数据的贝叶斯后验区间。

任务C：使epsilon不超过0.05至少需要多少n？样本数必须向上取整。若M扩大十倍，写出公式中哪一项发生变化，不要把半径直接乘十。

## 3 从数组看见“同一好事件”

```python
import numpy as np  # 提供数组与向量化比较。
from generate_data import population, sample  # 固定总体和IID抽样。
from experiment import true_risks, empirical_risks, radius  # 三个不同对象。
p = population()  # 包含全部候选在21格点上的预测表。
idx, y = sample(100, 7400)  # 有放回抽输入，再独立抽标签。
r = true_risks(p['predictions'], p['prob'])  # 精确总体风险，不是估计。
r_hat = empirical_risks(p['predictions'], idx, y)  # 经验风险随样本变化。
best = int(np.argmin(r_hat))  # 并列取第一个候选，规则事先固定。
eps = radius(100, 23, .05)  # 控制所有预定候选的最大偏差。
print(best, r_hat[best], r[best], eps)  # 不要交换训练和总体两列。
print(np.max(np.abs(r_hat-r)) <= eps)  # 本次样本是否处于一致好事件。
```

predictions形状为[23,21]；idx长度为100，值域为0到20；predictions[:,idx]变成[23,100]。把y扩为一行以后，不等比较同时计算每条规则在每个样本上的错误，再沿样本轴平均。若把axis写错，你会得到每个样本的候选平均错误，问题已完全改变。

## 4 独立重算重复结果

```python
import json, numpy as np  # 读摘要和未汇总数组。
from pathlib import Path  # 明确输出目录。
r = json.loads(Path('my_run/results.json').read_text())  # 不从图估数字。
a = np.load('my_run/trials.npz', allow_pickle=False)  # 全部数值记录。
for row in r['rows']:  # 每个n分开报告，不能混为同一次保证。
    values = a['n'+str(row['n'])]  # 400行，每行一次独立抽样结果。
    failures = int(np.sum(values[:,0] > row['epsilon']))  # 首列为最大绝对偏差。
    print(row['n'], failures, values[:,1].mean(), values[:,2].mean())  # 分别训练和总体。
```

固定结果应出现n=20组1次超界，其余组0次。若结果不同，先核对随机种子公式、NumPy版本及候选表，不要为了“通过”修改失败定义。使用严格大于号对应讲义中的坏事件；浮点等号边界在其他例子可需明确容差规则。

## 5 在同一数据上选表为何不能叫M=1

代码memorize把每个格点出现的标签多数类别记下来；未见或并列取0。亲自数出n=20中哪些格点没见过，解释默认0对正半轴总体风险的影响。比较同一次样本中阈值ERM与记忆表的训练风险及真实风险，不跨不同样本挑最好结果。

练习：事后仅保留这张表，代入M=1会得到多小的半径？指出证明第一步的哪个“固定”条件已经失效。再用全部2²¹张标签表作预定集合计算有效半径。不要把“错误半径比较小”写成更先进的方法；那只是一个没有当前证明支持的数字。

扩展方案：另生成一批独立评价样本，在模型冻结后用单规则界评价。如果再据此修改模型，须另留最终评价集或做新的选择校正。此扩展需独立实现并报告种子，发行结果不声称已执行它。

## 6 练习与交付

1. 写出单规则Hoeffding到有限类界的两步推导，解释因子2与M从哪里来。
2. 从一致界推ERM的2epsilon，明确两次转换分别作用于谁。
3. 400次中有1次超界是否反驳delta=0.05？0次是否证明失败概率为0？
4. 对每个n，画原始最大偏差分布与理论半径；禁止只画漂亮均值。
5. 写清这个界没有覆盖的两种数据依赖操作和一种分布变化。

交付手算、完整输出、图和一段有条件结论。结果解释至少区分均值、上界和尾部概率。答案见answers.md；Notebook使用同一run入口重做全部1600次抽样，实际顺序执行输出已保留，浏览器操作仍需学习者自行验证。
