# 007 代数函数与图像语言

先修：[001](../001/README.md)、[003](../003/README.md)。本单元解释方程、标量函数、坐标图、指数、对数、求和与复合；不要求NumPy、向量、矩阵或微积分。

## 学习顺序

1. 阅读[讲义PDF](lecture.pdf)，手算主要例子并画点。
2. 按[独立实验指南](lab.pdf)运行[Notebook](experiment.ipynb)或脚本。
3. 完成讲义9题，再对照[完整详解](answers.pdf)。

Markdown可编辑源分别为lecture.md、lab.md、answers.md。8幅原创建模与函数图位于figures，绘图源码为make_figures.py。

## 运行

在本目录运行：

```bash
python experiment.py
python test_experiment.py
```

只需Python3.12标准库，离线、小CPU、无需账号、API、训练数据或GPU。测试包含11组检查，含100组独立Fraction根值核验。脚本从任何当前工作目录调用都使用本目录的data/input_grid.csv，并把结果写入本目录outputs。

Notebook需要JupyterLab、ipykernel、nbformat；可按environment.yml建独立环境，然后启动JupyterLab，打开experiment.ipynb并重新启动内核后运行全部单元格。Notebook有实际顺序执行输出；构建端以新Python进程内的真实InProcessKernel验证，未声称验证Jupyter浏览器UI或套接字传输。

## 预期输出

- 方程2x+1=7得到3；0x+3=3为全部实数；0x+3=4无解
- 3厘米与30毫米经相应系数转换后都预测7；错误沿用系数得到61
- A=[1,9]、B=[4,4]在原均值与均值后log2下均选B，逐项log2再平均选A
- f(g(2))=9、g(f(2))=25；逐项加1求和为15，原和加1为13

outputs/function_values.csv保留非正输入的对数空白与outside_real_domain标记，避免用0冒充未定义值；results.json记录各项核验，environment.json记录实际Python版本。所有数据是课程原创合成记录。

## 重建图和PDF

核心学习不用重建。重建8幅图另需Matplotlib与Noto Sans CJK字体；make_figures.py只用Python列表调用绘图接口，不要求学生掌握NumPy。Matplotlib自身的底层依赖与课程先修不是一回事。

从课程仓库根目录执行：

```bash
python units/007/make_figures.py
python shared/build_pdf_mathjax.py units/007/lecture.md
python shared/build_pdf_mathjax.py units/007/lab.md
python shared/build_pdf_mathjax.py units/007/answers.md
```

PDF构建使用课程共用本地依赖与MathJax，不是学习实验运行条件。源码、官方来源核验与实际验收说明见source-checks.json、verification.json。数学定义域与计算机范围分别检查；本实验不声称处理所有浮点稳定性问题。未测试全新Anaconda安装或跨操作系统运行。
