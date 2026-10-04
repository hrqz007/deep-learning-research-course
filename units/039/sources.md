# 039 来源及实际核对范围

查阅日期：2026-10-04。下面只保留实际打开并核对的一手论文或官方文档。正文、推导例子、合成数据、机制图与代码均独立编写；不复制论文图，也不为外部资料附加新的版权许可。

1. Charles Elkan. The Foundations of Cost-Sensitive Learning. IJCAI 2001, pp.973–978. 作者公开PDF：https://cseweb.ucsd.edu/~elkan/rescale.pdf 。核对二分类条件期望代价、阈值与类别比例变化的讨论。本讲重新从两种行动风险推导，并显式要求固定代价与可信部署概率；不把本文历史经验观察当本实验结果。
2. Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. On Calibration of Modern Neural Networks. ICML 2017, PMLR70:1321–1330. https://proceedings.mlr.press/v70/guo17a.html ，原文PDF：https://proceedings.mlr.press/v70/guo17a/guo17a.pdf 。核对第4.2节共同正温度、验证NLL拟合、argmax保持，以及第5节校准评价。本讲的β凸性推导、边界检查与合成实验为独立展开，不声称复现论文大规模实验。
3. Zachary Lipton, Yu-Xiang Wang, Alexander Smola. Detecting and Correcting for Label Shift with Black Box Predictors. ICML 2018, PMLR80:3122–3130. https://proceedings.mlr.press/v80/lipton18a.html 。核对label shift定义：类别边际变化而X给定Y的条件分布不变。这里只引用问题条件，未实现或验证BBSE算法。
4. Rebecca Roelofs, Nicholas Cain, Jonathon Shlens, Michael C. Mozer. Mitigating Bias in Calibration Error Estimation. AISTATS 2022, PMLR151:4036–4054. https://proceedings.mlr.press/v151/roelofs22a.html ，原文PDF：https://proceedings.mlr.press/v151/roelofs22a/roelofs22a.pdf 。核对分箱ECE定义、有限样本偏差和箱数的作用。本讲报告5/10/20等宽箱敏感性，不声称实现该文偏差修正方法。
5. Juozas Vaicenavicius, David Widmann, Carl Andersson, Fredrik Lindsten, Jacob Roll, Thomas B. Schön. Evaluating model calibration in classification. AISTATS 2019, PMLR89:3459–3467. https://proceedings.mlr.press/v89/vaicenavicius19a.html ，原文PDF：https://proceedings.mlr.press/v89/vaicenavicius19a/vaicenavicius19a.pdf 。核对校准条件期望与评价定义差异。本讲明确采用正类概率分箱，与top-label confidence ECE区分。
6. PyTorch2.7 BCEWithLogitsLoss官方文档：https://docs.pytorch.org/docs/2.7/generated/torch.nn.BCEWithLogitsLoss.html 。核对logits输入、数值稳定组合、pos_weight含义、按元素数的mean归约。本讲硬标签等价性用实际torch2.7.1+cpu再次数值核验。
7. PyTorch2.7 CrossEntropyLoss官方文档：https://docs.pytorch.org/docs/2.7/generated/torch.nn.CrossEntropyLoss.html 。核对类别索引目标时加权mean的权重和分母，以及概率目标的不同公式。本讲只用两个logits的硬标签例子验证差异，不外推到所有轴和目标类型。

适用边界：理论总体最优不等于有限网络拟合；label shift修正不涵盖任意条件分布漂移；校准不保证任务代价下降；ECE的有限估计不是总体校准证明。正文中引用来源的目的为解释可核查条件，不以引用代替本讲代码和数值检查。
