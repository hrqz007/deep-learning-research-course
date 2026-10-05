# 一手来源与本单元使用范围

访问核对日期：2026-10-05。以下文献用于概念和API核对，本单元没有复制其图、代码、数据或实验分数。所有实验数值来自outputs/results.json。

1. Geirhos等，Shortcut Learning in Deep Neural Networks，Nature Machine Intelligence，2020，arXiv后续修订版：https://arxiv.org/abs/2004.07780 。用途：解释标准评价下有效但条件变化下失效的捷径现象。本讲的角标实验是独立教学设计，不是该论文的复现。
2. Gebru等，Datasheets for Datasets，CACM，2021，初始预印本2018：https://arxiv.org/abs/1803.09010 。用途：数据动机、组成、采集与适用范围的文档结构。不能据此替代某一实际数据集的许可审核。
3. Koh等，WILDS: A Benchmark of in-the-Wild Distribution Shifts，ICML，2021：https://proceedings.mlr.press/v139/koh21a.html 。用途：真实应用域变化需要明确独立域评价。本讲未下载WILDS，不声称复现其结果。
4. PyTorch 2.7 Reproducibility：https://docs.pytorch.org/docs/2.7/notes/randomness.html 。用途：控制随机数和确定性设置，明确跨平台与版本不保证逐位复现。
5. PyTorch 2.7 CrossEntropyLoss：https://docs.pytorch.org/docs/2.7/generated/torch.nn.CrossEntropyLoss.html 。用途：分类输入是未归一化logits，类别索引与reduction的语义。本单元使用默认均值归约。
6. Creative Commons CC0 1.0原文：https://creativecommons.org/publicdomain/zero/1.0/legalcode.en 。用途：本单元原创合成数组及元数据的数据释放说明；不覆盖第三方依赖与引用论文。

学习顺序建议：先完成微型账本与复现实验，再读捷径论文和WILDS的问题设置；阅读数据卡文献时，把每一项与本包的实际字段对照。引用他人结论时要保留其任务和数据条件，不把本例的17.71个百分点提升归到参考论文。
