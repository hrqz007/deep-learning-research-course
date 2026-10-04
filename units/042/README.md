# 042 可靠神经网络阶段项目

先修040。围绕重复测量数据建立一份从任务定义、完整反传到可信比较的MLP研究包：按行平均与按实体平均是否在优化同一个部署风险？

## 阅读与运行

- [正文PDF](lecture.pdf)及[Markdown](lecture.md)：两种风险、七参数两轮完整反向和第三前向、25参数训练、强基线、预算、实体区间与失败。
- [独立实验指南](lab.pdf)及[Markdown](lab.md)：运行、数据字段、数值文件、检查步骤、提交评分。
- [练习详解](answers.pdf)及[Markdown](answers.md)：12题完整推导和数值。
- [已执行Notebook](experiment.ipynb)、[训练脚本](experiment.py)、[测试](test_experiment.py)、[绘图](make_figures.py)、[数据生成器](generate_data.py)。
- [数据说明](data/README.md)、[一手来源](sources.md)、[验证范围](verification.json)。

按environment.yml建立环境，在单元目录执行：

```bash
python test_experiment.py
python experiment.py --output rerun
python make_figures.py --output rerun-figures
```

默认输出在outputs，显式相对路径按当前工作目录解释。脚本从任意工作目录可用完整路径调用。Notebook从单元目录或仓库根启动；第一格先核对源文件、输入、数值结果与图，再导入实验模块。随后在临时目录真正重做全部12次拟合并核对六份结果；静态图展示另由绘图脚本完整重建核验。

CPU float64，实际版本Python3.12.14、PyTorch2.7.1+cpu、NumPy2.3.5、Matplotlib3.10.8。绘图需Noto Sans CJK；可用DL_CJK_FONT指定字体文件。数据和代码不下载模型、不调用付费API。未验证新Anaconda跨平台安装、GPU、浏览器Jupyter UI或跨进程socket传输；Notebook在新进程内的真实IPython InProcessKernel执行。

## 完整证据与结果

40训练实体224行、32验证实体190行、160测试实体788行。两个25参数MLP条件各2学习率×3种子×350步，合计4200次更新、4212参数状态。两个条件匹配神经训练预算，经典求解器成本另列。全部候选与轨迹保留，无早停或剔除种子。

测试实体half-MSE：行加权MLP0.140537，实体加权MLP0.090165，三次多项式ridge0.052475。主差值-0.050372，按160个独立测试实体配对bootstrap区间[-0.078493,-0.025097]。实体加权在左区域有所退化；强基线更好。这是固定合成问题上的有限结果，不是普遍优势或因果机制证明。

六份数值文件包含完整预测、实体损失、精确手算、独立身份反例及全轨迹。training_traces.json.gz为固定mtime=0、空内部filename的无损gzip；results记录原文长度与SHA。先完成全部序列化再逐文件原子替换，不声称整个输出目录具备多文件崩溃事务。输入域、失败保护、测试数量和最终PDF摘要见verification.json。

作者完成检查、独立验收、远端公开是不同阶段，实际状态以课程发布记录为准。
