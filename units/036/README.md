# 036 归一化与训练推理状态

同一三行13参数模型完成两次BN前向、跨样本反向、全部参数更新与第三次前向；再比较LN轴、运行统计、模式和真实训练中的错误验证。

## 阅读与文件

- [正文PDF](lecture.pdf) · [可编辑正文](lecture.md)
- [独立实验指南](lab.pdf) · [实验源码](lab.md)
- [练习详解](answers.pdf) · [答案源码](answers.md)
- [已执行Notebook](experiment.ipynb) · [自包含实验](experiment.py) · [独立参考测试](test_experiment.py)
- [数据说明](data/README.md) · [来源及阅读范围](sources.md) · [实际核验与限制](verification.json)
- [两步完整计算链](outputs/hand-trace.json) · [轴比较](outputs/axes.json) · [模式和buffer](outputs/state-modes.json)
- [800次批组成采样](outputs/batch-sampling.csv) · [统计变化](outputs/batch-sensitivity.json)
- [全部六次训练](outputs/training.csv) · [全部轨迹及最终状态](outputs/training-detail.json) · [微批反例](outputs/microbatch.json)

先修018、030、035。公式、数据和代码均在本讲自包含，不跨单元导入。这里是有限CPU机制研究，没有归一化算法的普遍性能排名。

## 实际运行与复现

实测Linux CPU，Python3.12.14、torch2.7.1+cpu、NumPy2.3.5、Matplotlib3.10.8，float64、单线程。Anaconda声明供学习者使用，没有声称全新Anaconda或跨平台安装已验证。

```bash
conda env create -f environment.yml
conda activate dl-unit-036
python experiment.py --output my-run
python -m unittest -v test_experiment
jupyter lab
```

独立Python3.12环境也可按官方CPU渠道安装：

```bash
python -m pip install numpy==2.3.5 matplotlib==3.10.8 jupyterlab ipykernel nbformat
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

[PyTorch2.7.1官方安装记录](https://pytorch.org/get-started/previous-versions/#v271)。这里的版本是复现约束，没有声称最新推荐。无需GPU、torchvision、下载数据、账号或付费API；脚本不联网。

Notebook需重启内核后顺序执行；制作时在全新Python进程的实际IPython InProcessKernel中执行全部代码格。浏览器Jupyter与socket传输未验证。9份数值输出已在脚本、新进程-O和Notebook中重新计算并核对。make_figures.py独立重建11张原创图，需要本地Noto Sans CJK字体，路径不同时按本机安装修改。现成PDF无需重建图。

## 哪些结论受支持

- BN的跨样本Jacobian、全部局部路径和13参数两次更新有完整数字及独立Fraction/Decimal/自动微分核对
- LN由normalized_shape决定尾部统计轴；相同shape不保证相同归一化含义
- train/eval与grad/no_grad是不同开关；冻结参数不等于冻结buffer
- 正确微批权重无法消除训练BN统计组变化
- 六次训练各512次样本呈现，但B4为128次更新、B16为32次，不能作为等更新或等时间排名
- 错误train+no_grad验证会污染运行统计；污染后分数可能更好或更差，不能以方向判定协议正确

固定合成拆分，三种seed只改变重排，无测试排名、调参或总体显著性结论。9份产物是确定性教学数据；480次更新由独立NumPy重建。原始fixture及图/Notebook读入报告均受SHA256保护，防止改了输入仍套用旧图。探索新配置请写自己的文件与输出目录。

所有计算与序列化先完成，之后逐文件原子替换；不是整个目录断电事务。没有GPU、分布式同步BN、混合精度、跨版本或任意网络保证。state_dict例子只检查内存参数/buffer与模式语义，不是完整训练断点恢复演示。
