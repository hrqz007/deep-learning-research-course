# 002 Anaconda与Python第一份实验

先修：001 深度学习问题与证据。所有预测任务、单位、样本、候选、未命中规则与001一致。本单元从零解释Python，不使用NumPy或其他数值库。

## 阅读顺序

1. lecture.pdf：13页连贯正文，6张分布在对应章节的原创机制图。
2. lab.pdf：独立安装与操作指南，含三种系统分支、六个错误练习和Notebook重跑。
3. experiment.py或experiment.ipynb：完整核心实验。Notebook已经顺序执行并保存输出。
4. answers.pdf：手算、状态追踪、错误诊断与变化实验详解。

三个PDF均有同名Markdown原稿。figures与make_figures.py提供所有图及可重建脚本。数据为原创合成教学数据，无真实个人或设备记录。

## 最短运行路线

在已安装Python 3.12的环境中，在本目录执行：

```text
python experiment.py
python verify_experiment.py
```

核心脚本仅用标准库sys记录版本，其余为基本Python语法。脚本内置的数值列表由核验工具与data中的CSV逐项对照；没有外部文件、网络或相邻单元的运行依赖。输入适用范围是本例有限、对齐的数值；mae显式拒绝空列表和不同长度，完整通用输入验证留给后续单元。

从任意当前目录也可用本机完整文件路径运行脚本。verify_experiment.py会自行定位随附data和错误练习，不依赖当前目录。不要把error_examples内故意失败的文件当成核心实验损坏。

## Anaconda与Notebook

lab.pdf给出官方安装路径、系统差异、版本与解释器位置核对。特别说明：截至2026年10月4日，Anaconda官方说明已经停止为Intel Mac构建新包，旧版兼容性需另核查。这里没有宣称在所有系统上完成实机安装。

手动创建环境：conda create --name dl002 python=3.12，然后conda activate dl002。脚本路线无需额外包。需要Notebook时执行conda install jupyterlab ipykernel，再从本目录运行jupyter lab。

也可改用conda env create --file environment.yml一次创建，二者是替代路线。环境文件只表达版本系列和工具意图，不是逐平台锁文件。检查当前软件源、下载清单与条款提示，自行决定是否接受；不要通过关闭证书验证或安全软件排除安装错误。

## 预期结果

候选偏移量0、1、2的训练MAE依次为1.2、0.2、0.8，选择1。规则A训练MAE 0.2、测试MAE 0；规则B训练MAE 0、测试MAE 7；关系改为3x+1后，冻结A的压力测试MAE为3。

outputs/expected-result.txt省略平台相关的Python版本行，方便核对数值；outputs/actual-result.txt是本次实际脚本输出。相同数据重跑可以验证程序，不证明这些合成关系能用于真实设备。

## 核验材料与限制

- test-result.json：作者编写的43项实际检查，包括精确手算表、CSV对应、两次新进程输出、变化与同分案例、六个故意错误、空输入和长度不一致。
- notebook-check.json：在全新Python进程中使用真实IPython InProcessKernel，12个代码格顺序执行并保存输出；没有错误输出。
- source-checks.json：实际查阅的一手官方资料、日期、核查用途与访问限制。
- verification.json：PDF页数、文件摘要和本次核验范围。

作者自测不等于独立审稿。构建端运行平台是Linux、Python 3.12.14；没有进行Windows或macOS实机安装验证。构建环境不允许套接字传输，Notebook使用真实进程内内核；没有测试Jupyter浏览器界面或外部内核通信。

## 作者图像重建

仅重建图时需要matplotlib与支持中文的字体；这不是学生计算实验的依赖。安装已授权的绘图环境后执行python make_figures.py。默认查找Noto Sans CJK SC；其他机器可将CJK_FONT环境变量设为可用的中文字体文件。未找到字体会明确失败，避免静默生成缺字图。

PDF由课程共享Markdown与MathJax构建器生成，经WeasyPrint输出并逐页渲染检查。临时页面、依赖缓存和私人核查文件不在本单元交付目录。
