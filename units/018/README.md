# 第018单元 期望方差与抽样误差

先修[017 分布密度与似然](../017/README.md)离散部分。连续选做另外需要[010 导数积分与局部变化](../010/README.md)及017连续部分。本讲回答为什么一个小批次或一次运行不能代表全部情况。

## 学习顺序与文件

- [正文11页](lecture.pdf)：期望、方差、协方差、均值抽样分布、SE、复制组、MCSE、LLN与CLT边界；9幅原创图、14题
- [独立实验3页](lab.pdf)与[全部详解3页](answers.pdf)，对应Markdown源随附
- [Notebook](experiment.ipynb)：14个实际执行代码格；[脚本](experiment.py)与[11组测试](test_experiment.py)
- data包含本地合成配置与8组固定手算数据；outputs保留一次完整脚本结果，含32000批记录及8行汇总
- environment.yml为建议环境，source-checks.json和verification.json说明核验内容与限制

## 运行

核心脚本与测试仅用Python标准库，不需NumPy、GPU、网络或付费API。在本目录运行：

```bash
python experiment.py --output outputs
python test_experiment.py
```

建议Anaconda/Jupyter流程：

```bash
conda env create -f environment.yml
conda activate dl-unit-018
jupyter lab
```

在本单元目录启动内核，重启并运行全部。脚本可从任意工作目录调用，其默认配置相对脚本位置查找；Notebook的CSV与图路径相对本单元目录。脚本覆盖指定输出目录的四个文件，Notebook覆盖notebook_outputs；想保留旧实验请用新的--output目录。仓库只保存一套脚本结果及Notebook内嵌结果，不重复保存notebook_outputs。

可选python make_figures.py重建9图，需要matplotlib、NumPy和可用Noto Sans CJK字体。图与Notebook的手算、标签和文案固定对应默认配置，运行前同时检查data/config.json和outputs/results.json中的配置；不符就拒绝，且不改写图文件或notebook_outputs目录。自定义模拟请复制配置另存，并用脚本--config与--output写入独立目录；固定教学图和Notebook不会自动适配自定义设置。重建PDF从课程根执行python shared/build_pdf_mathjax.py units/018/lecture.md，再对lab.md和answers.md重复，见[共享构建说明](../../shared/build-tools/README.md)。重建后须重新目视检查所有页面。

## 关键结果

- 4个等权原子值0、0、2、6，E[X]=2、E[X²]=10、Var(X)=6
- n=64 IID均值方差3/32，真实SE≈0.30619
- 8个独立组各复制8次，64行真实方差3/4，SE≈0.86603
- 固定8组数据均值2，s²=48/7，正确估计SE²=6/7；复制64行的错误逐行估计SE²=2/21
- 4000批模拟，n=64两机制均值的经验SD约0.30820与0.87756；理论与模拟差异不是程序应强行消除的误差
- Bernoulli(p=0.001)、n=100全零概率约0.90479；样本SE为0不证明总体无不确定性

输入边界显式限制教学规模：精确数只收int、Fraction或分数字符串；非负整数质量须总和为正；样本方差需要至少2个观察；复制数、独立组数与CPU预算均检查。坏配置在创建输出目录或改写旧文件前拒绝。

已实测Python3.12.14新进程脚本、异目录python -S、14格真实InProcessKernel；四输出逐字节一致。11组测试包括150个展开总体、1360个IID序列、1008个复制组序列、100个协方差模型、精确n=64卷积、26个数学输入拒绝、12个配置拒绝和5段文档代码。17页最终PDF及9幅图逐页检查。

未实测全新Anaconda安装、Jupyter浏览器UI、跨进程socket内核和跨平台环境。有限模型与假设采样机制不证明真实数据独立、真实模型有效或泛化。独立审核与远端发布状态由根目录清单管理。
