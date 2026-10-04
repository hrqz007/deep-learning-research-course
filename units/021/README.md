# 021 指标拆分与可信评价

中心问题：一个数字怎样成为与任务目标相关的证据？

先修 005、018、020。本讲从合成检修告警出发，把混淆矩阵、分类分母、阈值选择、宏微平均、回归单位、概率校准与分组拆分连成同一评价过程，不预设梯度、优化算法或深度学习框架。

## 文件导航

- [正文 PDF](lecture.pdf)与[Markdown](lecture.md)，11 页，含 8 幅原创彩色图
- [实验指南 PDF](lab.pdf)与[Markdown](lab.md)，4 页
- [完整详解 PDF](answers.pdf)与[Markdown](answers.md)，5 页，覆盖 14 道正文题及实验 A 至 D
- [已执行 Notebook](experiment.ipynb)，12 个顺序代码单元
- [核心脚本](experiment.py)、[独立测试](test_experiment.py)、[环境建议](environment.yml)
- [数据字典](data/README.md)、[训练记录](data/train.csv)、[验证记录](data/validation.csv)、[封存测试记录](data/sealed_test.csv)、[拆分面板](data/split_panel.csv)
- [来源核查](source_checks.md)、[结构化来源记录](source-checks.json)、[作者验证范围](verification.json)

先阅读正文，再完成实验手算，最后对照详解。所有数据均原创合成，无真实个人或机构数据。sealed_test.csv 是教学封存约定，文件本身可读；提前查看会失去练习中首次独立评价的体验。

## 最小运行

核心计算与独立测试仅用 Python 标准库。从本单元目录运行：

```bash
python experiment.py
python -m unittest -v test_experiment.py
```

默认写入 outputs 下四个文件：frozen_plan.json、summary.json、thresholds.csv、calibration.csv。该目录随附的是实际执行结果，重跑会覆盖；使用 --output-dir my_run 可另存结果。脚本支持从其他当前目录以完整脚本路径启动，数据位置相对于脚本。

输入必须长度一致且非空；二分类标签是整数 0/1，bool 不算标签。分数与概率必须有限且在 [0,1]。多分类必须显式提供完整类别列表。回归接受绝对值不超过 1e100 的有限数值，拒绝会抹掉非零平方差的下溢情形；常数目标或单条记录的 R² 返回 null 及原因。空 precision/recall/F1 分母按 zero_division 填充，并保留 undefined 标记，不能把填充值当作观察证据。

## Notebook 与可选环境

```bash
conda env create -f environment.yml
conda activate dl021
jupyter lab
```

打开 experiment.ipynb，重启内核后从头运行。Notebook 将四份结果写到 notebook_outputs，并检查与随附 outputs 逐字节一致。notebook_outputs 是重跑副本，不属于发布文件。安装环境可能需要联网获取包，实验运行本身不访问网络。

作者实际使用 Python 3.12.14、ipykernel 7.4.0、nbformat 5.11.1，在新的 Python 进程中用真实 InProcessKernel 顺序执行 12 格并保存输出。environment.yml 是安装建议，不是已经验证的全新 Anaconda 安装或跨平台锁文件。浏览器 Jupyter 界面与跨进程 socket 传输未测试。

## 关键结果

- 验证集阈值 0.5：TP=2、FP=2、FN=1、TN=3，precision=1/2、recall=2/3、F1=4/7
- 事先规定漏报代价 3、误报代价 1，在五个候选中选阈值 0.4；验证平均成本 0.25
- 冻结后测试：8 条记录、3 个正例，TP=2、FP=2、FN=1、TN=3，平均成本 0.625、F1=4/7
- 三类示例的宏 F1=17/27、微 F1=10/13、支持数加权 F1=187/234；缺失类的处理会改变宏平均
- 回归 MAE=1、MSE=1.5、RMSE=√1.5、R²=0.7
- 独立校准例：四箱 ECE=0.1125，改为两箱后 ECE=0.025，Brier 均为 0.1925；换箱不等于概率改善

这些是有限合成计算，不是真实任务性能、显著组间差异或总体校准的证明。主表每实体两条记录，不能按独立八实体计算不确定性。

## 图与 PDF 重建

已有 figures 中的 8 张 PNG 可直接查看。make_figures.py 仅重建随附默认教学图；绘图库导入和写图前会核对原始四份数据与四份默认输出的摘要，拒绝混入自定义结果。自定义 --input-dir 实验必须用独立 --output-dir 保存。需要 matplotlib 和 Noto Sans CJK 字体。作者使用 matplotlib 3.10.8；两次新进程重建的 8 个 PNG 字节一致。

在课程仓库根目录运行：

```bash
python units/021/make_figures.py
python shared/build_pdf_mathjax.py units/021/lecture.md
python shared/build_pdf_mathjax.py units/021/lab.md
python shared/build_pdf_mathjax.py units/021/answers.md
```

PDF 工具依赖见[共享构建说明](../../shared/build-tools/README.md)，不属于核心数值实验依赖。

## 验证范围

13 组标准库测试覆盖 1,364 对二分类标签序列和 819 对三分类序列，以 Fraction 有理数算术独立核对计数指标与平均方法；另含阈值手算、缺失分母、回归边界、校准分箱边界、实体时间检查、31 项无效输入、输出目录与符号链接隔离。两个新进程脚本输出逐字节一致，并与 Notebook 四份输出一致。

流程测试在第一次读取测试文件时检查方案已写入，并证明改变测试标签和分数不会改变冻结方案。它是教学程序的执行路径核验，不是外部登记、权限封存或通用无泄漏证明。全部 20 页最终 PDF 与 8 幅图均已逐一目视检查。

当前状态为作者验收完成，独立课程 QA 与远端发布需以课程发布记录为准。未测试全新 Anaconda 安装、其他操作系统、浏览器 UI、socket 内核传输、GPU 或真实数据。构建中出现 Fontconfig 缓存权限和内核网络接口发现警告，实际产物已生成并核验，未把这些警告当作执行成功的替代证据。

独立复核修正：常数目标先按全部已验证的目标值识别，逐项检查平方及除以n后的下溢。每次运行在隔离尝试目录先写方案再读测试，成功后替换正式四文件；输入/计算失败保留旧结果，不声称抵抗操作系统写入失败的多文件原子性。
