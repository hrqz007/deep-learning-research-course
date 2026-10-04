# 030 模块数据加载与训练循环

怎样构成训练、验证、测试职责清楚，且可在完整epoch边界中断恢复的工程闭环？本讲用实际PyTorch CPU小网络验证参数注册、数据加载、损失归约、模式和状态管理。

## 阅读与运行入口

- [正文PDF](lecture.pdf) · [正文源码](lecture.md)
- [单独实验指南](lab.pdf) · [实验源码](lab.md)
- [完整练习答案](answers.pdf) · [答案源码](answers.md)
- [已执行Notebook](experiment.ipynb)
- [完整脚本](experiment.py) · [数值与故障测试](test_experiment.py)
- [数据说明](data/README.md) · [输出摘要](outputs/summary.json) · [来源记录](sources.md)

先修029 PyTorch张量与autograd、021 指标拆分与可信评价。代码自包含，不导入其他讲。数据72行全部合成；不需要GPU、网络数据、个人资料或付费API。

## 独立环境

实际验证：Linux x86_64、Python3.12.14、PyTorch2.7.1+cpu、NumPy2.3.5、Matplotlib3.10.8；float64、单CPU线程、确定性算法、num_workers=0。版本号是复现记录，不表示最新软件版本。

Anaconda学习者步骤如下，没有声称在全新Anaconda安装上实测：

```bash
conda env create -f environment.yml
conda activate dl-unit-030
python experiment.py
python -m unittest -v test_experiment
jupyter lab
```

也可先建立Python3.12的独立venv，在其中安装：

```bash
python -m pip install numpy==2.3.5 matplotlib==3.10.8 jupyterlab ipykernel nbformat
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

CPU安装命令见[PyTorch官方历史版本页](https://pytorch.org/get-started/previous-versions/#v271)。不需torchvision或torchaudio；macOS及其他平台应按官方平台说明选择，未实测跨平台安装。脚本不会自动安装软件或联网。

Notebook请从本讲目录打开，选同一环境内核，重启后顺序运行。实际验证使用新Python进程的InProcessKernel；未测试浏览器Jupyter界面和socket通信。首单元校验固定输入与提供输出，后续在临时目录重新计算；不要跳过首单元。Notebook内三张PNG已嵌入。

## 两种实际运行路径

```bash
python experiment.py --output my-run
python experiment.py --resume-checkpoint my-run/checkpoint-epoch-04.pt --output resumed-run
```

第二条是新进程恢复。将my-run/final-state.json与resumed-run/resume-final-state.json逐字节比较；本次验证完全相同。所有读取先验证，所有计算和序列化完成后才写结果；异常输入不会创建新报告或覆盖已有报告。最终替换是单文件原子操作，不承诺多个文件的断电事务一致性。

固定报告的三份输入有内置SHA256；字节变化会拒绝运行，防止新数据搭配旧图。用于探索的函数可在自己的新文件中调用，结果请另存，不改原始报告。restore是对本教学程序生成的可信检查点的恢复与一致性检查，不是处理任意恶意文件的服务。weights_only=True也不意味着未知来源.pt安全。

## 结果文件与关键证据

- first-step-trace.json：同一真实首批的全部中间量、局部链式梯度、17参数更新与更新后前向
- history.csv：12轮在线训练、轮末训练和验证误差，含错误批次等权对照
- orders.json：每轮45个训练ID，便于定位恢复差异
- checkpoint-epoch-04.pt：第4轮完整状态，可供真实新进程恢复
- final-state.json：第12轮完整状态的可阅读数值记录
- semantics.json：漏注册、浅快照、模式和动量手算探针
- test-predictions.json：验证选中第10轮后，对12行测试的预测
- summary.json：实际环境、预算、恢复与三种故意漏状态的差异

17参数，84步，540次训练样本呈现。最佳第10轮验证MSE约0.04587149，测试MSE约0.13905110；完整恢复相等。漏动量、全局RNG、加载器RNG的最终参数最大差分别约0.03149809、0.24162864、0.33410590。它们不是优化器优劣或真实任务性能结论。

12组测试包含完整首批17参数差分与逐量反向核验、60组独立NumPy前向、100组每组六步Fraction动量递推、九组种子与边界恢复、测试/验证干预、17种检查点损坏和输入失败保护。图重建用python make_figures.py，先校验3输入与8输出摘要再导入绘图库；需Matplotlib和本地Noto Sans CJK字体，字体路径可按本机情况调整。PDF正常阅读不需要重新构建。

## 验证范围

同一Linux CPU环境、固定dtype、单进程、完整epoch边界下的状态恢复；未验证GPU、跨版本/平台逐位一致、多worker、预取队列、中间批次、梯度累积、调度器、混合精度或真实数据表现。保存和比较逐位一致与科学结论可推广是两回事。全部证据范围见verification.json。
