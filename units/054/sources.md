# DL054 一手来源

核验日期2026-10-05。正文、数据、插图和数值例均为原创，不转载论文图文。

1. Devlin et al., BERT (2018). https://arxiv.org/abs/1810.04805 。区分原始MLM策略和本单元简化全MASK策略。
2. Raffel et al., Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (2019). https://arxiv.org/abs/1910.10683 。去噪与条件文本生成；未复现T5完整协议。
3. PyTorch2.7 CrossEntropyLoss. https://docs.pytorch.org/docs/2.7/generated/torch.nn.CrossEntropyLoss.html 。logits、整数标签、ignore_index及归约。
4. Lee et al., Deduplicating Training Data Makes Language Models Better (2021). https://arxiv.org/abs/2107.06499 。重复与训练评价的关系；本单元只检查规范化后精确重复。
5. Vaswani et al., Attention Is All You Need (2017). https://arxiv.org/abs/1706.03762 。encoder-decoder信息流。
