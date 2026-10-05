# 小型语言模型训练与规模实验

## 第055单元 独立训练和测量手册

本实验在CPU训练12个小型decoder，核对参数、token、计算和内存口径，再解释验证结果。先修033至035、054；目录所列065的测量知识已在lecture.pdf第六节补齐，无需等待另一单元。训练数据由本包生成，不连接外部语料、API或模型服务。

## 一 环境与执行

实测Python3.12.14、torch2.7.1+cpu、numpy2.3.5、Linux、单线程。每个配置用独立子进程测worker峰值RSS。建议从本单元目录运行：

```bash
conda env create -f environment.yml
conda activate dl055
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python test_experiment.py
python -O test_experiment.py
python experiment.py --output replay
python make_figures.py --results replay/results.json --output replay/figures
jupyter lab experiment.ipynb
```

测试会读取随包outputs的12个权重作验证，并另外跑一个8步小实验，不依赖网络。完整experiment.py在本机约42秒，但不是其他机器的承诺。Run All会再执行完整12配置，写notebook_replay，随后重画7张图。不要把重复执行误认为扩展新的超参数搜索；程序始终用同一协议。

每run固定80步、8960有效token；全12run共107520。每个worker训练循环步边界检查180秒，父进程240秒超时；超过上限报告未完成，不自动加时。CPU实测单worker峰值约312 MiB，Jupyter父进程和安装依赖还需额外资源，不能把该数字当整机内存要求。

已有Python3.12可用venv及requirements.txt。图与Notebook需要Noto Sans CJK；默认/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc。Debian/Ubuntu可安装fonts-noto-cjk；其他平台从Google Noto官方发行安装，并设置DL_CJK_FONT为真实字体路径。数值核心不依赖字体。resource测量只实测Linux；本课不提供Windows性能可比性保证。

已执行Notebook可离线阅读。create_notebook.py重建空版本；execute_notebook.py在新进程的IPython in-process内核执行，不验证浏览器UI或socket。可选PDF重建见build-tools/README.md，需Node、WeasyPrint系统依赖和Noto Sans/Serif CJK。

<div style="break-before:page"></div>

## 二 手算预算与优化

### 练习1 画完整训练样本

从data/corpus.json挑一条文档，按真实词表打印input和label，画7×7因果mask。验证正文六词元中的颜色2满足模4规则。解释为什么第0位置输入BOS预测颜色1、最后输入句号预测EOS，以及为什么没有padding时有效数是B×7。

### 练习2 数清每个参数

D=16、F=32、n=2、V=20、Lmax=7。分别数QKV、输出投影、FFN、两次块内LN、词嵌入、位置表、末端LN、词表头。推得5252，并对D=32推得18676。说明固定D改变头数为何不改变这里的参数量；共享输入输出矩阵会改变哪一项。

### 练习3 Token与重复度

B=16、L=7、80step时，求有效token、文档曝光次数、32与128文档池的平均重复次数，以及各自唯一目标位置数。说明为什么把8960写成“8960个新样本”会错。若有padding和梯度累积，应怎样重新定义有效数？

### 练习4 手算主要FLOPs

按乘加算2 FLOPs，从m×k乘k×n推到四类矩阵运算，再求两个宽度每步前向与估算训练代价。核对D16的74步与D32的20步预算差。写出被忽略的运算和为什么等FLOPs不是等时间。

### 练习5 一条真实AdamW路径

打开outputs/runs/d16_n128_s5501/results.json的前两个steps。用logit概率/标签公式解释head.bias[3]的原始梯度，按全部参数范数计算裁剪比例，再从m0=v0=0手算两步m、v、偏差校正和解耦衰减。比较记录中的before和after。不能把第二步梯度复用第一步，也不能把weight decay并入m的梯度。

<div style="break-before:page"></div>

## 三 运行并解释结果

### 练习6 计时和内存

查protocol、runtime及每个run结果。说明稳态吞吐分子为什么是8400，而总训练token仍是8960。训练计时包含/排除哪些过程？RSS为何远大于参数字节？GPU显存字段null是什么意思？不要在报告里把它改成0。

### 练习7 研究问题决定比较条件

完整重跑后，按宽度和池大小分组汇总第80步训练/验证NLL，保留3种子原值、均值和样本标准差。再只取大池128，对比D16第74步和D32第20步。解释固定token与近似固定矩阵计算量为何可以给出不同排序。

![复核图 这是等token结果中的训练验证差，不可直接解释为等计算效率。小池大模型的失败必须保留。](figures/04_gap.png)

### 练习8 平均指标掩盖了什么

绘出7个预测位置的验证NLL，找到固定格式和规则依赖位置。比较颜色2与ln4，说明为什么整体优于bigram不等于学会模4规则。推导完整均匀生成分布的4ln4/7，并说明为什么它不是有限训练/验证子集的严格经验loss下界。

### 练习9 能不能拟合缩放规律

只有两个宽度、两个固定数据池和三个初始化，能否可靠估计L∞、alpha、beta？给出欠缺的证据：留出规模点、更多数据划分、预算匹配、超参数优化控制和拟合残差。不要用两个点画直线就外推前沿模型。

<div style="break-before:page"></div>

## 四 验收与排错

提交9个练习答案、6类测试通过记录、完整12run结果、每组3种子原值、参数/FLOPs手算、两步AdamW复算、吞吐计时边界、RSS和GPU未测说明、逐位置指标和负面结果。原始测试划分不得用于挑超参数；本程序没有计算测试loss。

主要验收：参数5252/18676；每步112有效token；每run8960，总107520；唯一训练文档seen分别32/128；全部12权重回读后验证NLL与保存值相符；未来内容变化不影响更早logits；前缀截短在保持位置编号时数值一致；普通Python和-O均通过。

默认float32容差为1e-6量级；CPU版本差异可能导致舍入变化。测试不要求计时、RSS或ZIP字节在不同机器完全相同。吞吐噪声属于测量内容，应保留，不通过改线程或选择最快run来“修复”。

- loss不下降：先核对移位、有效目标、loss类别轴和参数是否真的更新，再查步长与梯度；不要立刻扩大模型。
- 训练好验证差：保留记录，查重复度、独立数据覆盖和分布；本课32文档大模型已给出这种反例。
- 梯度非有限：程序报错停止；不要静默nan_to_num后继续。确认输入、学习率和精度，定位后再定义新的实验协议。
- 计时大幅波动：先看CPU负载、进程启动、验证是否进了计时段及线程数。给出范围而非宣称稳定速度排名。
- RSS不随参数明显增加：固定解释器/库开销占主导，不能把RSS差当纯模型激活内存。
- checkpoint不能继续训练：npz只有模型权重，无optimizer、RNG、数据游标；本包只承诺读取评估或从头复现，不承诺无缝resume。
- 超过7个输入位置：本模型位置表和实验范围仅到7，明确拒绝；长上下文需要新的位置与训练协议。

权重读取必须使用np.load(...,allow_pickle=False)，按state_dict名字加载。不要加载不可信pickle。newdir保存新结果，公开包中不需要临时缓存或Notebook重跑目录。

一手来源和数据许可见sources.md、DATA_LICENSE.md。模型是合成文法教学实验，没有自然语言能力、实测GPU或前沿缩放外推结论。
