# DL076 实验：三个区间为何一起失效

本实验对应原大纲 T4。目标是在本地重新生成数据、真实训练五个网络、计算最后层后验并对比域内与域外区间。无需GPU、账号、PyTorch或在线数据。预计纸笔与分析60-90分钟，实际运行时间由设备决定。

## 1 环境与第一次运行

在本单元目录启动终端。首次安装可能访问软件包注册服务；安装结束后的实验完全离线。`requirements.txt` 是本次验证版本，`environment.yml` 给出可选 conda 环境。Python3.12是本次执行版本，其他平台未声称逐一安装测试。

```bash
# 安装运行与Notebook依赖；不会下载训练数据。
python -m pip install -r requirements.txt
# 普通模式运行数学/梯度/后验检查；预期7项通过。
python -m unittest -v test_experiment.py
# 优化模式仍须执行全部检查；预期同样7项通过。
python -O -m unittest -v test_experiment.py
# 另存你的完整实验，不覆盖发行版outputs证据。
python experiment.py --output my_run
```

命令中的 `-m` 表示把模块作为程序运行；`-O` 会删除普通 Python `assert`，所以测试使用 unittest 的断言方法，输入校验使用显式异常。`--output` 指定保存目录。预期得到 data.npz、weights.npz、predictions.npz、results.json。NPZ是压缩的多个数表；JSON是可用文本编辑器打开的结构化结果。

先检查 `results.json` 的三种方法与三个区域是否都在。单模型域内RMSE约0.184，远域覆盖0；不同底层数值库可能产生细微误差，不要求压缩文件逐字节与发行版相同。权重重载预测差应为0。若指标有大幅变化，先查是否改过种子、步数、数据或软件版本，不要直接“调到答案”。

### 读源码的正确顺序

先看 `make_data`：它只生成本地合成数据。再看 `forward` 和 `loss_grad`：前者给出网络输出，后者逐行写出均方误差的链式法则。接着看 `train` 的Adam更新，最后看 `features`、`last_layer` 和 `score`。训练函数没有评价标签参数，这是信息边界的一个可检查设计。

`train_x`有100项；隐藏表为100×16；最后层特征表为100×17；601个评价点的 `means` 为3×601，`members` 为5×601。先把这些形状写在纸上，能显著减少把“成员维度”和“样本维度”弄反的错误。

<div style="break-before:page"></div>

## 2 实验设计与纸笔核查

计划文件 [research_plan.json](research_plan.json) 固定训练支持、种子、预算和指标；它是本地设计文档，不是外部预注册。三种方法使用同一训练数据与评价网格。单模型复用成员11，集成保留五个成员，最后层方法使用成员11的已拟合特征。

**练习1：方差拆分。** 两个等权模型预测均值1、3，各自噪声方差0.04。写出混合均值、成员间方差和总方差。说明为什么混合方差分母用2而不是1。再问：若两者都预测3，真实均值却是0，成员分歧是否能发现共同错误？

**练习2：最小后验。** 只有一个常数特征，两个 y 为1和3，噪声方差1，先验精度1。手算 H、m、C，以及下一次观测的预测方差。区分潜在均值的方差与下一次观测的方差。

**练习3：均值后验与抽样。** 打开 `predictions.npz`，比较4000次抽样均值 `mc_mean` 与解析 `means[2]`。把抽样数改成40、400、4000，另存结果；记录最大差而不挑选有利种子。有限样本误差不保证每次随样本数严格单调下降。

```python
# 路径对象用于找到自己的实验数据。
from pathlib import Path
# NumPy读取本地压缩数组，不访问网络。
import numpy as np
# allow_pickle=False避免为普通数表启用对象反序列化。
p = np.load(Path('my_run') / 'predictions.npz', allow_pickle=False)
# 第一维是方法，2表示最后层Bayes。
error = np.max(np.abs(p['mc_mean'] - p['means'][2]))
# 输出此次Monte Carlo积分误差，典型量级是千分之几。
print(float(error))
```

**练习4：覆盖与宽度。** 从保存的 `test_y`、`means`、`variances` 手工复算 |x|≥4 区域的95%区间。使用1.96×标准差，不能把方差直接当标准差。分别报告三种方法的覆盖、平均宽度和RMSE；解释为什么更宽仍可能零覆盖。

**练习5：错误植入。** 在实验副本里把 `member.var(0)` 改成 `member.var(1)`，观察形状错误；再把观测噪声项删去，观察预测区间含义如何变化。不要修改发行版证据。每次记录改动、预期、实际及恢复方式。

<div style="break-before:page"></div>

## 3 Notebook、图与可审查报告

```bash
# 重建Notebook源结构；会清空此前保存输出，请先备份自写注释。
python create_notebook.py
# 新Python进程中顺序执行真实IPython内核，并保存输出。
python execute_notebook.py experiment.ipynb
# 从固定outputs读取数组重建六张原创图，不重新拟合。
python make_figures.py
# 仅修改文档后需要以下构建步骤；阅读现有PDF不需要它们。
python build_pdf.py lecture.md
python build_pdf.py lab.md
python build_pdf.py answers.md
```

Notebook会另写 `notebook_outputs` 并完整重训，不覆盖发行版 `outputs`。执行器采用进程内内核；这验证代码顺序和变量依赖，浏览器交互不是本讲的验证范围。文档构建另需 [build-tools/requirements.txt](build-tools/requirements.txt)、中文字体和系统Pango组件，详见 [build-tools/README.md](build-tools/README.md)。

请提交一页短报告，包含：你试图检验的主张、三个方法的真实计算差别、区域定义、三项指标、一个反例和至少两个限制。图必须标注训练支持区间，95%带是观测区间还是均值区间，集成带是否为精确混合分位数。将单模型训练成本与集成总训练成本分开写。

**练习6：只有一个因素的扩展。** 任选改变先验精度或训练支持范围，保留原报告再另存结果。提前预测哪部分会变、为什么；运行后如实写出不符预期的现象。禁止通过试多个设置只展示最好曲线，再声称是确认性结果。

**练习7：审查一个错误句子。** “4000次后验采样已经收敛，说明模型在未知地区可靠。”找出两个逻辑跨越，并改写成由本课证据支持的说法。额外讨论五个模型没有分歧时应不应该自动接收预测。

评价标准不按谁的RMSE最低，而按证据是否准确：30%可重跑与数据角色清楚，30%方差/后验手算，25%覆盖与失败机制，15%明确近似范围及预算。参考答案给出完整推理，但建议先独立完成练习1至4。

常见故障：缺中文字体会产生方框；它不会改变实验数值，却使图不可交付。不要只设置一个不存在的字体名字。图生成器显式注册本地Noto字体，可设置 `COURSE_CJK_FONT` 指向其他有中文覆盖的字体文件。安装失败应保留真实错误，不伪造“通过”。
