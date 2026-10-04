# 037 来源与核查记录

核查日期2026-10-04。正文、手算、合成数据、代码与13张图为本讲独立制作，没有复制论文图或商业教材。以下材料仅支持对应概念与框架约定；它们的基准结果不作为本讲的实测成绩，不为引用作品添加新的版权许可。

1. Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR15, 2014. https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf 。实际读取全文提取页2至6及模型定义、训练反传、边缘化讨论入口。原文以保留概率p描述未做inverted缩放的训练形式，本文重新定义q为保留概率，使用PyTorch的训练期缩放；两套记法没有直接混用。本文16掩码实例为原创精确计算。
2. PyTorch2.7 Dropout文档。https://docs.pytorch.org/docs/2.7/generated/torch.nn.Dropout.html 。实际读取p为丢弃概率、训练1/(1-p)缩放、评价恒等映射的条目。实际运行torch2.7.1+cpu的nn.Dropout核对模式、梯度和无运行buffer。主训练显式mask是同一公式，不承诺与官方内部RNG序列逐位相同。
3. Loshchilov and Hutter. Decoupled Weight Decay Regularization. ICLR2019, arXiv1711.05101v3. https://arxiv.org/abs/1711.05101 。本讲实际读取作者摘要及版本信息，关于普通SGD与自适应方法不等价的进一步公式沿用已核查的034，并在本讲独立推导固定D反例边界。https://docs.pytorch.org/docs/2.7/generated/torch.optim.AdamW.html 实际读取框架算法、参数组和weight_decay约定，训练设置在代码中显式指定。
4. Prechelt, Lutz. Early Stopping - But When? 作者公开稿。https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf 。实际读取15页PDF提取及停止准则、验证波动和效率效果权衡内容。只作为早停规则需明确的来源，本讲patience20规则与实际停止轮为独立实现，不借用论文成绩。
5. Chen, Dobriban, Lee. A Group-Theoretic Framework for Data Augmentation. https://arxiv.org/pdf/1907.10905 。实际读取第6页联合分布不变性定义、第28页监督条件/联合不变性的关系以及第36至38页近似不变性的偏差方差限制。正文只使用任务不变性观点，不声称满足该文所有定理条件。本讲两个镜像的标签机制通过已知合成定义直接核对。
6. Szegedy等. Rethinking the Inception Architecture for Computer Vision. https://arxiv.org/pdf/1512.00567 。实际读取第7节标签平滑，含softmax交叉熵logit导数p−q、均匀混合q′=(1−epsilon)delta+epsilon/K与KL方向说明。本文α=.1二分类目标.95/.05与单logit最优推导为独立展开。
7. PyTorch2.7 CrossEntropyLoss。https://docs.pytorch.org/docs/2.7/generated/torch.nn.CrossEntropyLoss.html 。实际读取类别索引/概率目标、归约、label_smoothing向均匀分布混合的定义。代码使用float64软目标独立计算并与框架核对；修复了作者首次检查中由默认float32软目标引入的约5.3e-10误差，不将其掩盖为理论差异。
8. Müller, Kornblith, Hinton. When Does Label Smoothing Help? NeurIPS2019. https://proceedings.neurips.cc/paper_files/paper/2019/file/f1748d6b0fd9d439f71450117eba2725-Paper.pdf 。实际读取摘要及第3节校准实验、蒸馏限制讨论。本文未把论文经验性改善写成保证；用平滑总体目标最优值和本次Brier配对结果说明边界。
9. PyTorch2.7 no_grad与Module.eval文档。https://docs.pytorch.org/docs/2.7/generated/torch.no_grad.html 及 https://docs.pytorch.org/docs/2.7/generated/torch.nn.Module.html#torch.nn.Module.eval 。实际读取模式切换与新计算图记录的区别。Dropout评价恒等返回原输入的requires_grad边界由本版本实跑验证，不单靠对文档一句话的过度概括。

## 证据层级

精确代数、分数计算和16张掩码枚举属于给定假设下的机制证据；CPU训练与图属于固定合成任务和有限设置的实测；源论文提供概念与原始定义。三者没有混作一个普遍有效性证明。没有外部实测数据、付费API、大型预训练模型下载或真实领域效果主张。
