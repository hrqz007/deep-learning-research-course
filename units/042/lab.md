# 042 独立实验指南

完成一份能从空会话重跑的重复测量MLP研究包，并清楚限定结论。

## 1 开始前写下三项判断

先读正文第2节，独立回答：一行是不是一个独立实体？部署是抽一行还是先抽实体？如果把某个实体的全部记录复制一遍，目标风险应不应该改变？本实验的答案分别是“不一定”“先抽实体再抽行”“实体风险不变”。这些决定先于模型名字。

本讲有固定参考实验和你自己的后续课题两层。参考实验用相同文件重现所有数值；后续课题允许改条件，但必须新建协议与评价数据，不能覆盖参考结果后仍声称它通过了原有哈希验收。所有数据为原创合成，无真实个人信息。

## 2 环境与完整运行

推荐依次阅读lecture.pdf、运行本指南、完成answers.pdf中的练习。已存在课程环境可以使用本单元environment.yml所列版本。若需要独立环境，在本单元目录运行：

```bash
conda env create -f environment.yml
conda activate dl-unit-042
python test_experiment.py
python experiment.py
python make_figures.py
jupyter lab
```

默认CPU float64，Python3.12、NumPy2.3.5、PyTorch2.7.1 CPU。实际制作使用Linux Python3.12.14；没有验证所有操作系统上的全新Anaconda安装、GPU、Jupyter浏览器界面或跨进程socket传输。Notebook在新Python进程内的真实IPython InProcessKernel中逐格执行；这与声称测试了浏览器UI不同。

绘图另需Matplotlib3.10.8和Noto Sans CJK字体。Linux默认检查标准字体位置；其他系统可将DL_CJK_FONT环境变量设为实际字体文件路径。缺字体会明确停止，避免悄悄生成方框字。阅读现成PDF不要求读者安装字体。数据生成、训练和绘图不访问在线服务。

脚本按自身位置定位data。可以从无关工作目录用完整脚本路径运行。显式相对输出路径则按当前工作目录解释。默认输出覆盖本讲outputs中的对应参考文件；建议先在新目录重跑：

```bash
python experiment.py --output rerun
python make_figures.py --output rerun-figures
python generate_data.py --destination regenerated-data
python -O test_experiment.py
```

实验只有12次神经拟合，每次350步，但程序会保留全部参数状态。不要把完成12次训练称为只运行了一个种子，也不要只复制Notebook留下的旧数字作为复跑证据。

## 3 认识数据与协议

train.csv有40个实体224行，validation.csv有32个实体190行，test.csv有160个实体788行。前两列是行ID和实体ID；replicate是实体内重复编号。x1和x2是唯一模型输入，y是目标。region由原始实体中心决定，用于预定分组报告，不能按测量x1的符号重新划分。

protocol.json规定目标、拆分、两种损失权重、两个学习率、三个种子、步数、正则、基线和bootstrap。fixture-hashes.json记录输入摘要；experiment.py还固定关键fixture摘要，修改数据或协议后默认参考运行会拒绝。若要开展新实验，复制整个单元并记录新的协议，不要通过删除检查来掩盖数据变化。

第一项程序练习是检查实体不重叠并核对权重。不必先读取测试标签。

```python
import numpy as np
import experiment as e
train = e.load_split('train')
validation = e.load_split('validation')
if set(train['groups']) & set(validation['groups']):
    raise RuntimeError('实体跨拆分重叠')
q = e.weights(train['groups'], 'entity')
print(len(train['y']), len(set(train['groups'])), q.sum())
print('每个训练实体权重:', sorted({
    round(sum(q[i] for i, name in enumerate(train['groups'])
              if name == group), 12)
    for group in set(train['groups'])
}))
```

应得到224行、40个实体、权重和约1，每个实体总权重0.025。左区域一个实体有8行，每行权重1/320；右区域有2行，每行权重1/80。行加权则所有行都是1/224。两种训练模式使用同一份训练实体加权均值和标准差。

## 4 先核对三行七参数的完整链

Notebook会真正调用Fraction计算器，输出两轮的每个前向量、残差、权重上游、三个样本对七参数的贡献、惩罚导数、总梯度与更新；随后计算第三次前向。初始数据损失0.4296875，含正则目标0.5171875。两次更新后的总目标分别约0.362900942672与0.295774145026。

手算时至少逐项完成第一例的两条隐藏路径，然后把其余两行加上。第二轮不能复制第一轮梯度；必须重新使用$a=(0.911875,-0.504375)$。查明为什么A2单行误差变差而总目标下降。若你的总梯度相差一个3或一个2，先检查是否对已归一化权重重复求平均。

公开测试对两轮共42个逐样本参数贡献使用autograd交叉核对，另在8个随机平滑网络上检查全部25个参数，共200个中心差分坐标。有限差分不替代推导；通过一次随机点也不能说明所有输入正确。

## 5 真正重跑与保留每一步

experiment.py实际执行两种目标、两个学习率、三个种子，共12次拟合4200次更新。所有运行都完成既定350步，没有早停或事后剔除。每轮保存更新前状态，末尾再保存完成350次更新的状态，所以共有4212个状态，不是4200个。

|产物|如何使用|
|---|---|
|results.json|协议、所有候选分数、选参、预算、基线及摘要|
|hand_chain.json|精确分数与浮点显示的纸笔账本|
|training_curves.csv|每步训练目标、实体风险、验证风险、梯度范数|
|training_traces.json.gz|每次神经拟合的完整参数与曲线，无损保存|
|final_evaluation.json|所有固定预测、种子结果、实体损失和bootstrap|
|identity_leakage.json|独立身份记忆反例的原始观测与推导数字|

读取无损轨迹并检查第一次更新，不要求手动解压出大文件：

```python
import gzip, json
import numpy as np
import experiment as e
with gzip.open('outputs/training_traces.json.gz', 'rt') as f:
    runs = json.load(f)['runs']
r = runs[0]
t = e.load_split('train')
x = e.transform(t['x'], r['normalization'])
q = e.weights(t['groups'], r['mode'])
p = np.asarray(r['parameters'][0])
g = e.numpy_gradient(p, x, t['y'], q, r['l2'], r['width'])
np.testing.assert_allclose(p-r['lr']*g, r['parameters'][1],
                           atol=2e-15, rtol=1e-13)
print('运行数:', len(runs))
print('状态数:', sum(len(z['parameters']) for z in runs))
```

gzip固定mtime=0、空内部filename和level9，并在results中保存未压缩长度及SHA。当前固定环境比较压缩字节；若压缩器版本改变，要同时比较解压原文，不能把封装差异自动解释成数值变化。

## 6 检查公平比较与选择

每个MLP条件预算都是2100次更新、470400行样本呈现，25参数。相同种子的两个目标从逐位相同的参数开始。候选率0.03和0.1都保留三种子；按各自平均最终验证实体风险选择，两者都选0.1。

验证按行模型0.086825、按实体模型0.092875的排序，与最终测试主结果不一样。不要把验证中不利于主假设的数字删掉。平均种子损失用于选择，平均预测用于部署，二者明确区分。

强基线包含常数、线性ridge和十系数三次多项式ridge。后者四个λ候选全保留，由验证选0.01。请在报告中列出四次多项式、一次线性、一次常数拟合，不能把这些额外成本隐藏起来，也不能宣称不同求解器墙钟时间相同。

## 7 解释最终评价与失败

先核对160个实体、788行，主指标始终是先实体内平均再对实体平均。两MLP主指标0.140537与0.090165，差约-0.050372。2000次整实体配对bootstrap的95%百分位区间约[-0.078493,-0.025097]。三次多项式为0.052475，强于两MLP。

报告至少包含三个限制：左区域实体MLP从0.044943退化为0.050049；强基线更好；区间只涵盖固定训练和选参后的测试实体抽样，不含重训练与重新选择。至少复查一个正差值实体和一个负差值实体，解释总平均不能代表每个实体。

身份反例是另一个纯记忆任务。先前三行、测第四行的损失约0.006828；训练60个身份、测试20个新身份约0.426475。不能把它当成本次主MLP的性能，也不能混同两个不同部署任务。

## 8 输出保护与输入边界

本讲数值接口拒绝布尔、字符串、复数、非有限数与转成float64后溢出的扩展精度数。训练输入和目标限制绝对值不超过一百万；这只是教学实现的计算域，不是统计假设。近常数训练特征会明确报错，避免除以几乎零的标准差。

脚本先验证fixture和输出路径，再训练。全部数值序列化成功后才写文件；逐文件原子替换不是跨文件事务。不要将输出设置为源码、data、figures或已有文件。测试失败时先查看实际错误，不要删除检查、改参考摘要或放宽容差来掩盖差别。

Notebook第一格在导入实验模块前核对关键源文件、输入、数值输出与图；从单元目录或课程根启动。所有代码格顺序执行，临时目录重新跑全部12次训练并比较六个文件。图格展示冻结图，make_figures.py另行重建并核对，不把静态图片冒充实时训练。

## 9 提交要求与后续课题

交付一份简短研究报告及可复跑包。任务与目标风险20分，完整手算和梯度验证20分，强基线与预算20分，拆分和条件区间15分，失败分析与结论15分，文件组织和复现证据10分。缺少独立测试单位、删掉不利结果或用测试选参，不能仅靠模型分数弥补。

报告应明确哪些结论由程序直接核对，哪些是待检验解释。随后可选择一个新问题，例如把每实体重复次数统一，或在新数据上比较更多训练实体与更多重复测量的作用。先写新协议、分配新验证和测试数据，再实验；这些是新研究提案，本单元没有替你执行或声称其结果。
