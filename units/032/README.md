# 032 小批量随机梯度下降

以固定参数点的精确抽样、可解标量噪声递推、同一13参数网络的完整更新和明确预算对照，解释噪声梯度的条件与代价。

## 阅读与文件

- [正文PDF](lecture.pdf) · [源码](lecture.md)
- [独立实验指南](lab.pdf) · [源码](lab.md)
- [完整答案](answers.pdf) · [源码](answers.md)
- [执行后的Notebook](experiment.ipynb)
- [数值脚本](experiment.py) · [独立标量参考与测试](test_experiment.py) · [独立计时](benchmark.py)
- [数据说明](data/README.md) · [预算对照](outputs/budget-comparison.csv) · [同一首批完整追踪](outputs/first-batch-trace.json)
- [来源](sources.md) · [验证范围](verification.json)

先修011偏导梯度与链式法则、018期望方差与抽样误差、031梯度核验与最小调试法。无需导入其他讲代码。32行是全部合成训练数据，本讲没有独立测试集或真实任务性能排名。

## 运行

实际Linux CPU环境：Python3.12.14、PyTorch2.7.1+cpu、NumPy2.3.5、Matplotlib3.10.8；float64、单线程。版本为复现记录，不表示最新软件。

下列Anaconda为学习者说明，未声称新安装实测：

```bash
conda env create -f environment.yml
conda activate dl-unit-032
python experiment.py --output my-run
python -m unittest -v test_experiment
python benchmark.py --output my-timing.json --repetitions 5
jupyter lab
```

如使用独立Python3.12 venv，在该环境安装：

```bash
python -m pip install numpy==2.3.5 matplotlib==3.10.8 jupyterlab ipykernel nbformat
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

CPU包选择见[官方历史版本页](https://pytorch.org/get-started/previous-versions/#v271)。不需要torchvision、GPU、模型下载、私人数据或付费API；脚本不自动安装或联网。macOS等平台按官方指引选择，未实测跨平台安装。

Notebook从本目录打开并重启内核顺序运行。构建实际使用新Python进程的InProcessKernel；浏览器界面和socket通信未测。首单元检验3份输入与8份原始输出。七份确定性结果会在临时目录重新计算并比对；时间单独重新观测，不要求与旧秒数相等。嵌入三张PNG图已核对文件字节。

## 实验协议与结果

- 精确抽样：四个二维梯度，B=1—4；有放回协方差Sigma/B，不放回(N−B)Sigma/[B(N−1)]；全相关重复不会缩小方差
- 条件偏差：两行随机重排例，给定历史后的下一梯度与当前全梯度不同
- 标量模型：均值、方差、噪声平台及eta=2边界，明确限定独立有放回假设
- 小网络：2→3→1 tanh，13参数，主目标为平均半平方误差；同一ID25/9/6/4包含完整前后向、13梯度、更新、下一次前向
- 三种预算：相同1280样本呈现、相同40步，以及相同1280呈现但学习率=.02×B/4；保留同一预生成ID流

固定eta=.02、1280次样本呈现时，B=1/4/16/32的训练目标约.0108465/.0428222/.0701057/.1385584；线性缩放后都约.0428。这里只显示比较协议的影响，没有全面调参、初始化重复或泛化结论。

## 数值与计时分开

experiment.py生成summary.json、budget-comparison.csv、trajectories.json、sampling-exact.json、scalar-theory.json、network-sampling.json、first-batch-trace.json七份确定性结果。measured-timing.json是另外实际运行benchmark.py的一次观测批次，包含预热、五次原始纳秒值、中位数、范围、脚本摘要和测量窗口；默认数值脚本不覆盖它。

计时包含整个train(record=False)调用的校验、索引展开、Tensor初始化与前后向/更新/保护；不包含导入、数据读取、风险日志和写文件。CPU共享负载影响秒数；不外推GPU或大型网络吞吐。重新计时请给新输出文件，避免新数字与原图混配。

输入及绘图固定报告均有SHA256保护；变化后明确拒绝。探索新数据/参数请在自己的新脚本里调用函数并另存结果。所有计算和序列化完成后才写结果；最终替换仅单文件原子，不提供多个输出在断电情形下的整体事务。图重建python make_figures.py需要本地Noto Sans CJK字体，可按本机调整脚本字体路径；正常学习无需重建PDF。

9组测试包括80个独立标量前后向模型、13坐标差分、224个精确抽样协方差场景、全部短IID路径矩核验、所有12条主训练轨迹逐步独立重建、失败输入保护。详细边界与实际执行证据见verification.json。
