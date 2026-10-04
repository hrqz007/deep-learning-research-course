# 019 熵交叉熵与KL散度

中心问题：分类损失、分布匹配和生成目标为何常出现对数？

## 学习顺序

先修017与018的离散主线：概率质量、对数似然、有限期望和抽样误差。正文显式补充证明所需的最小导数工具，并给出几何检查，不预设Jensen不等式或连续概率。所有主计算使用有限分布，不涉及GPU训练。

1. 阅读正文，手算主例与零支持，再证明KL非负和等号条件
2. 按实验指南运行脚本及Notebook
3. 对照详解完成14道练习，解释受限模型族、方向与经验分布的区别

## 文件导航

- [正文PDF](lecture.pdf)与[Markdown](lecture.md)，11页
- [实验指南PDF](lab.pdf)与[Markdown](lab.md)，4页
- [完整详解PDF](answers.pdf)与[Markdown](answers.md)，4页
- [执行后的Notebook](experiment.ipynb)、[核心脚本](experiment.py)、[独立测试](test_experiment.py)
- [合成数据说明](data/README.md)、[分布CSV](data/distributions.csv)、[固定标签CSV](data/labels.csv)
- [来源核查](source_checks.md)、[结构化来源记录](source-checks.json)、[作者验证范围](verification.json)

figures/含8幅原创彩色机制图；make_figures.py可重建。outputs/含已实际运行的summary.json、directional.csv和epsilon.csv。Notebook保留全部14个代码单元的真实输出，重跑时另外写notebook_outputs/，该重复目录和任何缓存均不作为发布文件。

## 最小运行

核心实验和独立测试只用Python标准库。从本单元目录运行：

```bash
python experiment.py
python -m unittest -v test_experiment.py
```

也可从别的目录给出完整脚本路径。使用--output-dir可另存结果而不覆盖参考。代码允许1至64个状态，检查概率范围和状态数，只对总和误差不超过1e-12的舍入级输入明确归一化。合法的支持不匹配返回正无穷，非法概率或标签拒绝。正α的平滑份额下溢会拒绝；CSV原始表头须精确、唯一，每行字段完整；所有必需分布、计算和序列化完成后才创建或改写结果，输入失败保留旧输出。标准JSON用字符串“Infinity”保存无穷。

Notebook需要ipykernel、nbformat与可用Jupyter前端。建议Anaconda配置：

```bash
conda env create -f environment.yml
conda activate dl019
jupyter lab
```

重启内核后依次运行每一格。作者实际使用Python3.12.14、ipykernel7.4.0、nbformat5.11.1。environment.yml是建议配置，未完成全新Anaconda安装测试；不能据此声称在每个平台都已安装成功。

## 图与PDF重建

已有PNG可以直接查看。重建图需要matplotlib及可用Noto Sans CJK字体；make_figures.py会检查并注册字体。作者使用matplotlib3.10.8，已核对两次生成的8个PNG字节一致。当前构建环境产生Fontconfig缓存权限警告，但图和PDF已实际生成并逐一检查，不将警告掩盖为未执行。

在仓库根目录可运行：

```bash
python units/019/make_figures.py
python shared/build_pdf_mathjax.py units/019/lecture.md
python shared/build_pdf_mathjax.py units/019/lab.md
python shared/build_pdf_mathjax.py units/019/answers.md
```

PDF构建依赖见[共享工具说明](../../shared/build-tools/README.md)，不属于学习者核心数值实验依赖。

## 关键结果与验证边界

- 主例H、交叉熵、KL分别1.5、1.75、0.25 bit，nat值各乘ln2
- P=(1/2,1/2,0)、Q=(1,0,0)时，正向KL无限、反向ln2
- 受限三候选中，正向选宽，反向左右并列；将P加入后两个方向都选P，不支持一般模式寻求定律
- 罕见事件实验说明有限批次的经验0不排除总体交叉熵无穷

作者已完成11组测试：784个分母6的三状态有理分布对，由80位Decimal独立交叉核对；363条长度1至5的标签序列，由精确bit码长核对；零支持、极小正数、无效输入、语义反例与两次独立新进程复现。Notebook在新进程的真实InProcessKernel顺序执行14格，三份输出与脚本逐字节一致。全部19页最终PDF与8幅图均已目视检查，变更后的页面另行复核。

这属于作者验收，独立课程QA与远端发布由课程发布记录另行说明。没有测试浏览器Notebook界面、socket内核传输、全新Anaconda环境、其他操作系统、GPU性能、真实数据校准或真实生成质量。合成实验只展示有限机制。
