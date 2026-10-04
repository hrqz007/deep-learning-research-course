# 017 分布密度与似然

中心问题：模型输出概率或密度时，它究竟在假设什么？

## 学习顺序

- 第一部分只以016《随机变量与条件概率》为直接先修，学习有限Bernoulli分布、编号生成器、联合概率、似然、对数与端点、有限候选后验。
- 第二部分进入前必须先补010《导数积分与局部变化》，再学习密度的积分归一化、Gaussian位置/尺度、数值积分误差与连续似然。
- 不预设018期望和方差。所有数据为原创合成，不能当作真实系统表现的证据。

## 阅读与文件

- [正文PDF](lecture.pdf)与[可编辑Markdown](lecture.md)
- [实验指南PDF](lab.pdf)与[Markdown](lab.md)
- [练习详解PDF](answers.pdf)与[Markdown](answers.md)
- [顺序运行Notebook](experiment.ipynb)、[核心脚本](experiment.py)、[独立测试](test_experiment.py)
- [合成数据](data/observations.csv)及[data说明](data/README.md)
- [来源核查](source_checks.md)、[结构化核查](source-checks.json)、[作者验证范围](verification.json)

figures/包含9幅原创教学图，make_figures.py可用matplotlib重建。outputs/包含三份已实际运行的脚本结果。Notebook本身保留全部实际执行结果，重跑时另写notebook_outputs/，该重复输出目录不作为发布材料。

## 最小运行

核心脚本和测试只需Python标准库。先进入本单元目录：

```bash
python experiment.py
python -m unittest -v test_experiment.py
```

也可从别处启动完整脚本路径，数据按脚本自身位置定位。`--output-dir`可指定输出目录。输出为bernoulli.csv、gaussian_integrals.csv和summary.json。生成器用局部Random实例、整数seed=17和有限概率3/4；固定手算数据另存为1110，不由每次生成结果替换。

Notebook需ipykernel、nbformat和可用Jupyter前端。environment.yml提供建议Anaconda配置：

```bash
conda env create -f environment.yml
conda activate dl017
jupyter lab
```

先完成离散A段；补010后再运行B段。重启内核并从第一格顺序执行全部代码，切勿依赖上一次会话的变量。作者实际使用Python 3.12.14、ipykernel 7.4.0、nbformat 5.11.1与matplotlib 3.10.8；未执行全新Anaconda安装，上述环境文件不是安装成功凭据。

PDF可用课程共享的本地MathJax构建器重建。在仓库根运行：

```bash
python shared/build_pdf_mathjax.py units/017/lecture.md
python shared/build_pdf_mathjax.py units/017/lab.md
python shared/build_pdf_mathjax.py units/017/answers.md
```

所需PDF构建工具见[共享工具说明](../../shared/build-tools/README.md)，不属于学习者核心实验依赖。图片重建需可用Noto Sans CJK字体；脚本明确注册字体文件。

## 预期结果与限制

对1110，p=1/4、1/2、3/4的似然分别为3/256、1/16、27/256。先验为9:1时两候选后验各1/2。σ=0.1的Gaussian峰高约3.9894，区间[−0.1,0.1]概率约0.6826895。σ=0.01在[−1,1]上，20面板中点法漏峰，2000面板接近1。

作者测试覆盖9组、2142个精确序列似然、230个有限编号映射组合、21个高精度log参考、72个Gaussian点参考及4个独立积分参考，另有47种数值拒绝输入与7种非法CSV。CSV须恰有唯一的trial,y两列表头；重复y、重复trial与多余列在输出目录创建前拒绝，并检查已有三份输出不变。14个Notebook代码单元已在新进程中的真实InProcessKernel依次执行，三份结果与脚本逐字节一致。所有PDF页面均渲染后逐页目视检查。

这属于作者验证，独立课程审核和远端发布状态由课程发布记录另行说明。没有测试浏览器Notebook界面、socket内核传输、跨平台新安装或实际数据分布适合度。小实验展示机制，不提供真实任务的泛化结论。
