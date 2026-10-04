# 015 局部近似与优化几何

中心问题：曲率、步长和参数尺度怎样共同影响优化？

先修：[011](../011/README.md)、[013](../013/README.md)。本单元不依赖014。沿用行参数(1,2)、行梯度(1,2)和Hessian(2,2)。所有数据为原创无量纲合成参数，离线CPU即可运行。

## 学习材料

- [正文PDF](lecture.pdf)与[可编辑Markdown](lecture.md)：Taylor近似、积分余项、Hessian、凸性、驻点、精确稳定区间与尺度
- [实验指南PDF](lab.pdf)与[Markdown](lab.md)：手算、运行、观察、反例、失效输入与证据要求
- [12题详解PDF](answers.pdf)与[Markdown](answers.md)
- [已执行Notebook](experiment.ipynb)、[对应脚本](experiment.py)、[独立测试与Fraction oracle](test_experiment.py)
- [原创参数说明](data/README.md)、[配置](data/config.json)、[环境](environment.yml)
- [来源核查记录](source_checks.md)、[结构化来源](source-checks.json)、[作者验证记录](verification.json)
- figures/含10幅原创机制图；make_figures.py可重建。outputs/含脚本示例结果。

## 运行

在本单元目录执行：

```bash
conda env create -f environment.yml
conda activate dl-unit-015
python experiment.py
python test_experiment.py
jupyter lab experiment.ipynb
```

Notebook须从当前目录重新启动内核、顺序运行全部单元。它生成notebook_outputs，并与outputs逐文件比较。正式包只提供outputs与Notebook内的实际执行输出，不重复提供notebook_outputs。更改配置后请使用新输出目录，以免失败后误读上次成功的文件。

```bash
python experiment.py --config data/config.json --output my_run
```

主例H=[[13,-12],[-12,13]]，特征值1和25，初值(2,0)，损失26。固定全梯度步长对所有初值收敛的范围是严格的0<η<0.08。边界0.08的损失仍下降，却趋25；这是一项必须能解释的反例。程序只输出有限次轨迹，长期结论来自正文证明。

## 复建图和PDF

读者只运行核心实验不需要matplotlib或PDF依赖。重建图还需matplotlib 3.10.8及Noto Sans CJK字体；从仓库根目录运行：

```bash
python units/015/make_figures.py
python shared/build_pdf_mathjax.py units/015/lecture.md
python shared/build_pdf_mathjax.py units/015/lab.md
python shared/build_pdf_mathjax.py units/015/answers.md
```

PDF依赖和本地MathJax配置见[共享构建工具](../../shared/build-tools/README.md)。没有运行时在线公式渲染。

## 证据和限制

作者运行了新进程脚本、11组独立测试及真实InProcessKernel中的13个Notebook代码单元；三份PDF全部逐页渲染并目视检查。详见verification.json。此记录只代表作者验证，不宣称独立审查或远端发布已经完成。

未测试本轮全新Anaconda安装、跨平台安装、Jupyter浏览器界面、独立内核socket传输、GPU或真实神经网络。float64有限性检查不能消除舍入、相消和所有下溢；数学严格为0与显示为0必须分开。实验不支持把合成正定二次函数的结论推广为非凸网络全局收敛或泛化保证。
