# 038 容量过参数化与理论边界

从同一两样本的完整梯度链与零空间出发，解释最小范数选择、参数化改变、插值附近的噪声放大和理论条件。1944次CPU合成拟合保留全部seed，包括无双下降和更宽反而更差的对照。

## 阅读与产物

- [正文PDF](lecture.pdf) · [正文源码](lecture.md)
- [独立实验指南](lab.pdf) · [实验源码](lab.md)
- [12题练习详解](answers.pdf) · [答案源码](answers.md)
- [已执行Notebook](experiment.ipynb) · [完整Python实验](experiment.py) · [18组测试](test_experiment.py)
- [数据说明](data/README.md) · [来源与阅读范围](sources.md) · [实际核验记录](verification.json)
- [两轮完整手算链](outputs/hand-trace.json) · [零空间与坐标缩放](outputs/implicit-bias.json) · [6与11参数完整网络链](outputs/duplication.json)
- [全部1944次结果](outputs/capacity.csv) · [全部系数和训练预测](outputs/capacity-detail.json)
- [条件风险和Monte Carlo诊断](outputs/conditional-risk.json) · [谱滤波](outputs/spectral-filters.json) · [有限候选上界](outputs/finite-class-bound.json)

先修013、020、037。讲义、代码与数据在本目录自包含，不需要调用前一讲文件。

## 实际环境与运行

实测Linux CPU，Python3.12.14、NumPy2.3.5、PyTorch2.7.1+cpu、Matplotlib3.10.8，float64，单线程。无GPU、在线数据、账号或付费API。Anaconda声明给学习者参考，未声明全新Anaconda环境或所有系统均已安装验证。

```bash
conda env create -f environment.yml
conda activate dl-unit-038
python experiment.py --output my-run
python -m unittest -v test_experiment
jupyter lab
```

已有独立Python3.12环境时也可使用：

```bash
python -m pip install numpy==2.3.5 matplotlib==3.10.8 jupyterlab ipykernel nbformat
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

[PyTorch官方2.7.1安装记录](https://pytorch.org/get-started/previous-versions/#v271)。版本为本实验复现约束，不代表最新软件推荐。Notebook须重启内核后顺序执行；制作中使用全新Python进程内真实IPython InProcessKernel，没有测试Jupyter浏览器或socket传输。make_figures.py在已核验数据上重建12张原创图，需本机Noto Sans CJK字体；现成PDF不要求文档构建环境。

## 结论范围

- 两轮线性手算、零空间选择、重参数化，以及同一数据的6和11参数ReLU网络完整路径可逐项核对
- 18宽度×3噪声×3惩罚×12设计/噪声seed共1944个直接最小二乘/Ridge拟合，不是深网训练轨迹
- clean_risk是已知各向同性总体中对新输入的精确平方风险，noisy_risk再加新标签噪声方差
- 记录一次训练噪声实现与条件噪声期望的区别，不混淆中位数与期望
- 固定惩罚不是调参最优，没有真实任务性能或普遍双下降保证
- 通用逼近、线性GD偏置、有限类界、比例渐近各有前提，不能替代评价证据

9份数值产物在正常脚本、-O新进程和Notebook中重新计算；Notebook显示3张冻结PNG，全部12张图另行重建核验。18组测试包含1944次独立lstsq/增广Ridge参考、Fraction两轮、34个逐样本神经网络参数贡献与17个梯度。数据/图输入/首格均有哈希守卫。所有计算与序列化先完成，再逐文件原子替换；不声称整个目录断电事务。
