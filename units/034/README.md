# 034 自适应优化器与权重衰减

沿用033的两行九参数ReLU网络，完整追踪两次Adam更新；再检查epsilon、偏差修正、L2与AdamW的状态差异，最后执行一份包含全部候选的等预算比较。

## 文件与阅读顺序

- [正文PDF](lecture.pdf) · [正文源码](lecture.md)
- [独立实验指南](lab.pdf) · [实验源码](lab.md)
- [完整答案](answers.pdf) · [答案源码](answers.md)
- [已执行Notebook](experiment.ipynb) · [计算脚本](experiment.py)
- [独立标量与Decimal测试](test_experiment.py)
- [教学数据说明](data/README.md) · [全部搜索81行](outputs/search-grid.csv)
- [两次完整计算链](outputs/hand-trace.json) · [选优记录](outputs/selection.json) · [冻结测试](outputs/test-report.json)
- [一手来源及阅读范围](sources.md) · [实际验证与限制](verification.json)

先修033。所有代码自包含，无跨讲导入。主数据为80行合成记录，48训练/16验证/16测试；手算例子另有固定两行。Notebook中的旧输出供核对，仍应重启内核顺序运行。

## 实际运行环境

Linux CPU，Python3.12.14，PyTorch2.7.1+cpu，NumPy2.3.5，Matplotlib3.10.8；float64，单线程。以下Anaconda是学习者说明，未声称全新Anaconda或跨平台安装已实际验证：

```bash
conda env create -f environment.yml
conda activate dl-unit-034
python experiment.py --output my-run
python -m unittest -v test_experiment
jupyter lab
```

如果使用独立Python3.12 venv：

```bash
python -m pip install numpy==2.3.5 matplotlib==3.10.8 jupyterlab ipykernel nbformat
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

CPU安装路径见[PyTorch官方历史版本](https://pytorch.org/get-started/previous-versions/#v271)。版本是复现约束，不是最新版本推荐；其他系统依官方渠道选择，没有声明逐平台实测。无需GPU、torchvision、网络数据、付费API或模型下载；脚本不自动联网。

## 实验结论与边界

- 两步手算Adam：同一数据经过全部前向、局部导数、两行路径累加、九梯度、两个状态、修正、更新与下一次前向；损失.0221→.00518052347→.00004514480
- 独立二维给定梯度例核对耦合L2和AdamW，揭示它们连后续动量符号都可能不同
- 实测None与零Tensor、纯衰减乘积及只恢复参数造成的状态错误
- SGD+动量、Adam、AdamW各9候选×3重排顺序；每次96更新/768样本呈现。公开所有81行，没有删掉不佳候选
- 固定初始化，seed只控制重排；以最后一步三个顺序的平均验证半MSE选优，随后各做3次测试评价。图中的9次轨迹重放另计，非新增候选

三个家族都选中衰减0，故Adam/AdamW结果相同；SGD+动量选alpha=.03，二者选alpha=.01。不能据此宣称AdamW普遍最佳，也不能宣称L2与解耦一般等价。Adam类最佳点位于学习率网格边界，搜索并不充分；一个合成拆分和三个训练顺序不足以形成总体显著性结论。相同步数与样本次数不代表相同时间，这里不报告速度排名。

## 可复核与保护

八份确定性输出包括手算追踪、机制对照、搜索表、选择、已选轨迹、冻结测试、状态检查及摘要。正常脚本、无关工作目录中的python -O和新进程真实IPython InProcessKernel均实际执行并逐字节对照。Notebook浏览器界面与socket通信未测试。

固定三份输入被SHA256保护；图重建与Notebook首格另核对原始报告，避免变更配置后套旧图。要探索新设置，在自己的文件里调用公开函数，写新目录。所有计算与序列化在最终写入之前完成，替换是每文件原子操作，不是整个目录的断电事务。

python make_figures.py可重新绘制九幅原创图，需要Matplotlib和本机Noto Sans CJK字体；若字体路径不同，请据本机安装修改。学习现成PDF无需重建图或安装文档构建工具。恢复例子单独使用衰减.1作用于全部九参数，其余Adam参数沿用手算；仅验证内存state_dict深拷贝，不声明磁盘、跨版本、GPU、混合精度或分布式验证。
