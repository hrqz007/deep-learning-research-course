# 第020单元 经验风险与泛化

先修[018 期望方差与抽样误差](../018/README.md)。本讲回答训练平均损失为什么只是目标总体风险的替代；用有限分类问题区分拟合、模型选择、近似、估计、经验优化与分布变化。

## 文件与顺序

- [正文9页](lecture.pdf)，8幅原创图、12题；[独立实验3页](lab.pdf)；[完整详解3页](answers.pdf)，均附Markdown源
- [Notebook](experiment.ipynb)含14个实际执行代码格；[脚本](experiment.py)与[11组测试](test_experiment.py)
- data提供默认配置、固定4条手算训练数据与生成机制说明；outputs提供9600条模拟候选/选择结果、12行汇总、手算与环境
- 来源、作者核验范围与建议环境分别见source-checks.json、verification.json、environment.yml

## 运行与边界

核心只用Python标准库，不需GPU、NumPy、网络或付费API。在本单元目录运行：

```bash
python experiment.py --output outputs
python test_experiment.py
```

建议Anaconda/Jupyter环境：

```bash
conda env create -f environment.yml
conda activate dl-unit-020
jupyter lab
```

Notebook请在本单元目录启动、重启内核并顺序运行。脚本默认数据路径相对自身，支持其他工作目录。四份脚本结果会覆盖所指定输出目录的同名文件；Notebook写notebook_outputs。仓库只保留一套脚本结果和Notebook内嵌结果。

固定教学图与Notebook会检查原始默认配置、固定手算表及对应results.json，不符即在写图或生成Notebook结果前拒绝。自定义模拟使用--config、--training、--output另存；不要用固定图文解释不同设置。核心输入只收声明范围内的整数类别与标签，拒绝布尔值、非法质量、缺失/重复CSV字段与超预算配置。先完成数据检查、计算和序列化，再创建/改写输出，输入失败不破坏旧结果。

可选python make_figures.py需要NumPy、matplotlib和Noto Sans CJK字体，重建8图。重建PDF从课程根运行python shared/build_pdf_mathjax.py units/020/lecture.md，再对lab.md与answers.md重复，详见[共享构建工具](../../shared/build-tools/README.md)。之后仍需逐页检查。

## 关键结果

- X均匀取0到7，前四类P(Y=1|X)=0.1，后四类为0.9；最佳总体分类风险0.1
- 固定4条数据：常数模型训练/总体为1/2与1/2；阈值为1/4与2/5；查表为0与1/2
- t=4相比训练ERM，经验优化差为1/4，总体风险差为−3/10；不能把总体差硬称为非负优化误差
- n=8、800次重复，阈值/查表平均总体风险0.18675与0.29225；n=128时本次均约0.100125
- 验证集负责选择，独立测试只评价冻结规则；精确总体风险只审计，不参与拟合或选择
- 条件标签机制反转后，同一旧最优规则风险从0.1到0.9；仅输入质量改变时固定学得阈值风险为43/70

作者验证：256条规则对80联合原子；120个训练集独立枚举ERM；80个加权总体；40个训练集上全部9阈值的风险上界；平方损失精确联合核对；22个数学输入拒绝及15个坏CSV/配置保护场景。全部9600模拟记录的训练包含关系与验证选择另行核对，5段文档代码执行通过。脚本在新进程及异目录python -S运行，14格真实InProcessKernel执行，4产物逐字节一致；15页PDF、8图及3个Notebook图已查看。

未实测全新Anaconda安装、Jupyter浏览器UI、socket内核传输及跨平台安装。合成总体已知的审计特权不适用于现实任务；小实验不证明真实模型泛化或漂移机制。独立验收与远端发布状态由根目录清单管理。
