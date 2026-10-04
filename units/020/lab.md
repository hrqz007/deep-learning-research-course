# 经验风险与泛化实验

## 目标与先修

先修018。本实验比较同一有限总体上三个嵌套候选类，区分训练误差、验证选择、独立测试与精确总体风险。不要把“训练更小”自动翻译为“泛化更好”。核心仅需CPU和Python标准库；Notebook需要Jupyter。图已提供，重建图另需NumPy、matplotlib及中文字体。

先读data/README.md，明确X、Y的生成机制和损失。原始手算数据、默认配置及相应outputs共同构成固定教学材料，图和Notebook运行前会核对它们。自定义脚本可用--config、--training、--output，但必须另存，不混入固定图文。

## 一 环境与重跑

在本目录运行：

```bash
python experiment.py --output outputs
python test_experiment.py
```

需要新环境时可参考：

```bash
conda env create -f environment.yml
conda activate dl-unit-020
jupyter lab
```

在本单元目录启动内核，重启并运行全部Notebook。脚本默认路径相对自身，支持从其他目录调用。outputs包括results.json、trials.csv、summary.csv、environment.json；重跑会覆盖指定输出目录的同名文件。Notebook写notebook_outputs，交付只保留一套脚本结果及Notebook内嵌输出。

Anaconda新装、浏览器UI、跨进程socket内核与跨平台安装未实测。实际检验采用Python3.12.14新进程脚本与真实InProcessKernel。建议环境文件说明依赖，不保证所有平台安装成功。

## 二 先算总体再看训练

对每个x，预测0时错误率η(x)，预测1时错误率1−η(x)。写下总体最优$h^*$与其风险，再核对程序：

```python
from fractions import Fraction as F
from experiment import risk, empirical_risk, fit, candidates, load_training
bayes = (0,0,0,0,1,1,1,1)
assert risk(bayes) == F(1,10)
assert risk((0,)*8) == F(1,2)
```

读取四条手算训练记录，逐条标出三个规则错在哪，再运行：

```python
train = load_training()
models = {kind: fit(kind,train) for kind in ('constant','threshold','lookup')}
for kind,h in models.items():
    print(kind,h,empirical_risk(h,train),risk(h))
assert empirical_risk(models['lookup'],train) == 0
assert risk(models['lookup']) == F(1,2)
```

应得到常数训练1/2、总体1/2；阈值训练1/4、总体2/5；查表训练0、总体1/2。阈值t=1不是从总体风险挑出来的；拟合函数只看训练错误数，字典序处理并列。查表逐位置多数投票，未见或平票位置输出0。

## 三 检查优化和总体是否一致

```python
assert empirical_risk(bayes,train) == F(1,2)
assert risk(bayes)-risk(models['threshold']) == F(-3,10)
```

$h^*$在这批数据上训练更差却总体更好。写出经验优化差1/4，并解释为什么总体差−3/10不能当作一个必定非负的优化误差。不要由一个反例推出“越不优化越好”。

对阈值类枚举9个候选，分别画训练与总体风险。再算所有候选的最大绝对偏差：

```python
uniform_error = max(abs(empirical_risk(h,train)-risk(h)) for h in candidates('threshold'))
assert uniform_error == F(11,20)
```

近似误差为0、精确ERM的经验优化差为0，因此正文上界是2×11/20=11/10。它正确却很松；风险本来不超过1，不能把界大于1当概率。报告“公式没有提供有用的有限样本保证”比只展示一个符号更准确。

## 四 样本量与容量实验

默认对n=8、32、128各重复800次。每次三个候选由同一训练批产生，只依32条验证记录选一个，随后用256条新测试记录评价。总体真值只供审计。检查simulate中selected这一行仅读验证错误数，并列顺序预先固定，没有测试或总体风险参与。

n=8时阈值平均训练误差约0.08250、总体0.18675；查表约0.05672与0.29225。n=128两类总体风险在本次重复中均约0.100125。查表在每个相同训练批上的训练最优不差于阈值，应逐次检查这一点；总体排序不受这个集合关系保证。

查看summary.csv的mcse_population_mean：它是800个“训练后模型的精确总体风险”的平均的Monte Carlo标准误。不是现实一个模型的置信区间，也不是独立测试样本SE。说明每个随机层次的对象，比只报小数位更重要。

## 五 两种分布变化

先冻结bayes规则，只把条件概率反转：

```python
assert risk(bayes,(9,9,9,9,1,1,1,1)) == F(9,10)
assert risk(models['threshold'],weights=(1,3,3,3,1,1,1,1)) == F(43,70)
```

第一式改P(Y|X)，第二式仅改P(X)。两者都没有继续训练。请解释为什么源目标已精确最优，仍无法保证新目标风险低；这不等于可以只凭真实分数下降就判定发生哪种漂移。

## 六 失败路径与提交

尝试非法标签、布尔值输入、负质量、空训练集、重复CSV列名、缺失列和超预算配置。应在输出创建或改写前拒绝，保留旧结果。测试使用临时目录验证这些性质。已学到的“支持完整数据”也包括不静默丢弃一列。

重启Notebook顺序运行，四个结果文件与脚本逐字节比较。提交三项：固定数据的手算与枚举核对；三种容量的训练、独立测试与总体风险解释；目标分布、独立单位、选择规则、复现版本及不能推出的结论。完整练习详解另附，但先完成自己的误差来源分析。
