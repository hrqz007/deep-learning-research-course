# 来源与适用范围

核对日期：2026-10-08。以下为原始论文或官方文档；未复制外部图像、数据集或训练权重。教材中的数值、推导算例和图由本单元原创脚本生成。

1. [Guo et al. On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)

   用途与边界：温度缩放与现代网络校准；本课独立合成实验不声称复现论文基准。

2. [Ovadia et al. Can you trust your model's uncertainty?](https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html)

   用途与边界：分布变化下的校准和不确定性评估局限；本课数据与指标由原创代码产生。

3. [Geifman and El-Yaniv. Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500)

   用途与边界：风险与覆盖权衡的研究背景；本课未实现原文统计风险控制算法。

4. [PyTorch 2.14 Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

   用途与边界：随机种子、确定性算法和跨平台复现限制。

5. [PyTorch Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

   用途与边界：state_dict、weights_only与推理保存和续训保存的区别。

6. [PyTorch 2.14 CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)

   用途与边界：分类交叉熵直接接收logit与类别索引。
