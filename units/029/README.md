# 029 PyTorch张量与autograd

把027的相同数据、参数与损失迁移到实际PyTorch CPU，逐层比较数值与所有参数/输入梯度，定位原地修改、detach、重新构造tensor、梯度累加和eval误解。

## 阅读顺序

- [正文](lecture.pdf) · [可编辑Markdown](lecture.md)
- [单独实验指南](lab.pdf) · [实验源码](lab.md)
- [完整练习答案](answers.pdf) · [答案源码](answers.md)
- [已执行Notebook](experiment.ipynb)
- [程序](experiment.py) · [独立NumPy与标量前向参照](reference.py) · [测试](test_experiment.py)
- [原始教学数据说明](data/README.md) · [结果摘要](outputs/summary.json) · [来源](sources.md)

先修：004数组张量与形状思维、014浮点数与稳定数值计算、028自动微分机制与边界。固定案例来自027，程序自包含，不导入其他讲。本讲不需要GPU、模型下载、真实个人数据或付费API。

## 环境与运行

实际验证环境：Linux x86_64，Python3.12.14，NumPy2.3.5，PyTorch2.7.1+cpu，CPU float64；另外实际运行float32对照。版本号是复现记录，不代表最新版本。官方2.7文档与这组API相匹配。

推荐先建立独立环境，避免覆盖现有项目。下列Anaconda步骤是学习者安装说明，没有声称在全新Anaconda安装中实测：

```bash
conda env create -f environment.yml
conda activate dl-unit-029
python -c "import torch,numpy; print(torch.__version__, numpy.__version__)"
python experiment.py
python -m unittest -v test_experiment
jupyter lab
```

若使用独立Python3.12 venv，可在该环境中安装：

```bash
python -m pip install numpy==2.3.5 matplotlib==3.10.8 jupyterlab ipykernel nbformat
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

CPU命令来自[PyTorch官方历史版本安装页](https://pytorch.org/get-started/previous-versions/#v271)。本讲只需要torch，不需要torchvision或torchaudio。macOS轮子与Linux/Windows选择可能不同，需按官方页选择；未实测跨平台安装。脚本不会自动安装或联网。

在Notebook中选择相同环境的Python内核，重启后顺序执行全部14个代码单元。构建时实际使用新进程内的InProcessKernel，未验证浏览器Jupyter界面或socket内核通信。Notebook中的三幅图已嵌入；也保留外部PNG供重新执行。

从其他工作目录也可用绝对路径执行experiment.py。可用`--output`指定新的结果目录；`--data-dir`只接受两份固定教学输入的原始内容，任何字节变化先报错且不改旧结果。探索新数据请通过reference.py与experiment.py的函数，在新文件中构造输入，保留原始报告便于对照。

## 输出与检查

- `comparisons.csv`：197个前向/反向标量坐标，逐层最大差2.220446049250313e-16；包含每层H/Z、dH/dZ及26参数梯度
- `finite-difference.csv`：26参数加10输入坐标，独立标量前向中心差分最大误差约1.4594e-11
- `semantic-probes.json`：复制/别名/leaf、累加、原地失败、断图、模式、VJP和高阶图
- `tensors.json`：完整固定批次数值快照
- `summary.json`：实际版本、精度和摘要

损失约1.1489788656；一次学习率0.1更新后约1.1413798716。正确2+3微批加权误差约4.16e-17；错误等权约0.0330774。12组测试还含72个随机小网络的mean/sum逐层比较、18个额外全坐标差分模型、100个独立Fraction仿射层、21个多项式二阶导数点和失败保护。

图重建：`python make_figures.py`。重建前校验固定两输入与五输出摘要，修改后拒绝重画；需要Matplotlib和Noto Sans CJK，提供的Linux默认字体路径可根据本机字体位置调整。构建PDF需要课程共享的MathJax/WeasyPrint工具；正常学习无需重新构建PDF。

## 限制

这是数值与框架语义检查，不是模型性能、泛化或收敛证据。函数的中间仿射值绝对值限制为1e6；差分扰动若超出输入/参数边界会明确拒绝，不保证边界点可作中心差分。CPU与GPU、不同dtype或库版本不保证逐位一致；异常字符串也可能随版本改变。None与零梯度不同，非叶未retain_grad的None也不自动等于断图。eval和no_grad不同，身份映射模块直接返回已有输入时还会保留输入本来的requires_grad标志。

固定输入字节保护用于防止新数据混配旧图。所有输入、计算、序列化成功后才写结果；最终替换仅单文件级原子，不承诺操作系统故障下五文件整体事务原子性。新Anaconda安装、GPU、跨平台、浏览器Jupyter和socket传输未实测。完整检查范围见verification.json。
