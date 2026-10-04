# 041 迁移学习与适配

先修036与040。先读lecture.pdf，再按lab.pdf执行experiment.ipynb，练习解答在answers.pdf。所有文本同时提供Markdown源。

完整比较：源任务自训表示；两种目标任务、16/64训练标签、128验证标签、512测试样本、3个配对初始化；从零、随机冻结、预训练冻结、两种分组微调、探测后微调和RBF岭。三个源训练加252个目标候选，全量保存轨迹与预测。

在课程仓库根目录运行：

```bash
python units/041/experiment.py --output reproduced-041
python units/041/test_experiment.py
python units/041/make_figures.py --output reproduced-041-figures
```

运行Notebook前读lab.md的定位与完整性说明。原版数值结果位于outputs；results.json.gz是确定性gzip封装的完整JSON，summary.json保存压缩前后摘要，calculation细节在calculations.json。无需下载模型或数据，无付费API。

主要结论有明确边界：正交目标的小步微调在指定160步目标预算下出现负迁移，等步微调实际恢复；不能把冻结或小步失败推广为所有迁移失败。源到目标输入边缘分布改变，相关与正交目标共享输入、仅标签规则不同。

预算并非端到端等计算：源训练、目标更新、候选搜索和解析岭解分别报告。三次初始化共用同一数据拆分；RBF确定性结果不构成三份独立样本。没有真实任务、GPU或浏览器Jupyter效果保证。验证状态与限制见verification.json，来源检查见sources.md。
