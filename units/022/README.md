# 第022单元 比较实验与统计不确定性

先修[018 期望方差与抽样误差](../018/README.md)和[021 指标拆分与可信评价](../021/README.md)。怎样区分方法效果、随机波动、调参选择与错误的独立单位？

## 内容与交付

- [正文11页](lecture.pdf)：配对方差、区间覆盖、t假设、bootstrap、精确失败反例、实体/种子层次、多重比较、效果量与消融
- [独立实验4页](lab.pdf)、[完整详解4页](answers.pdf)，14题及A至D实验答案；Markdown源全部提供
- 9幅原创图，嵌入相关推导；[Notebook](experiment.ipynb)13个实际执行代码格，3张内嵌图
- [标准库脚本](experiment.py)、[10组测试](test_experiment.py)、合成种子表、计划、配置、五份结果及source-checks.json

## 运行

在本单元目录执行：

```bash
python experiment.py --output outputs
python test_experiment.py
```

可选环境：

```bash
conda env create -f environment.yml
conda activate dl-unit-022
jupyter lab
```

Notebook重启内核后顺序运行，另写notebook_outputs并核对五份文件。核心脚本仅标准库，支持不同工作目录；无GPU、网络、真实训练或付费API依赖。environment.yml是建议，未实测全新Anaconda、浏览器UI、socket内核和跨平台安装。实际运行Python3.12.14与真实InProcessKernel。

--config、--rows、--plan支持显式输入路径，但本教学实现只接受已声明的固定比较协议、八个0..7配对和整数百分数。配置限制见lab；输入错误先于计算/写入拒绝，所有计算和序列化先于输出目录创建。有效重跑覆盖同名五文件；自定义结果请用独立--output。无OS多文件写入故障原子性保证。

固定图/Notebook在开始时检查默认配置、主表、计划和对应results.json，拒绝混用合法自定义结果。make_figures.py可重建9图，需要NumPy、matplotlib及Noto Sans CJK字体。PDF构建见[共享说明](../../shared/build-tools/README.md)，输入本单元lecture.md、lab.md、answers.md；重建后仍须逐页检查。

## 已核对的关键数值

八个差异(2,4,0,6,−2,4,2,0)，平均2个百分点；样本方差48/7，配对SE约0.92582。误当独立两组会给4.60590。舍入t7区间约[−0.18956,4.18956]；正态近似约[0.18539,3.81461]，本例未证明正态差异条件，不能择优报告。

全部8^8经验bootstrap以整数卷积精确计数，条件方差3/4，percentile区间[1/4,15/4]。真实20×Bernoulli(0.01)、n=8反例的percentile真实覆盖约0.07720137，说明穷举经验重采样不保证名义覆盖。

四实体复制八行的正确/错误均值方差估计分别5/3、5/31。纯噪声选择例有6000行记录，独立复测期望仍为0；它不是p值模拟。实际重要性与等效结论均须预先定义，消融效果依赖上下文。

作者验证包括200配对有理数夹具、117短表的全部有序重采样、稀有分布256个真实序列与9×256重采样序列、8种候选数的精确最大值、28数学拒绝、24坏输入/配置新旧输出隔离调用、五文件新进程与Notebook逐字节对比。19页PDF和9图逐一查看；完整范围见verification.json。独立验收和远端发布状态以根清单为准。
