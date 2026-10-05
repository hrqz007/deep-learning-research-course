# 一手来源与证据范围

资料核对日期2026-10-05。推导、合成数据、图与测试为本课程原创；下面来源用于结构背景、实际版本接口与解释边界，不替代本课独立实验。

1. Vaswani等，Attention Is All You Need，2017，原作者论文。scaled dot-product结构与缩放动机；本课完整导数独立推导。https://arxiv.org/abs/1706.03762
2. PyTorch 2.7 SDPA官方版本文档。bool mask允许语义、scale与dropout参数。https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html
3. PyTorch 2.7 MultiheadAttention官方版本文档。投影层、attn_mask/key_padding_mask禁止语义、need_weights。https://docs.pytorch.org/docs/2.7/generated/torch.nn.MultiheadAttention.html
4. Jain与Wallace，Attention is not Explanation，NAACL 2019，原作者论文页。用于提醒注意力权重不自动等于解释，不把其特定任务结论扩为一切模型。https://aclanthology.org/N19-1357/

全屏蔽行NaN/零输出并非根据公式或摘要推测，而是在PyTorch 2.7.1+cpu、float64、单头identity投影、无dropout、need_weights两个值下实际调用记录。跨设备或版本未测。

系统字体来自Google Noto CJK官方项目或Linux官方软件包。本包不附字体二进制。https://github.com/notofonts/noto-cjk

可选PDF构建的官方安装与依赖见shared/build-tools/README.md。授课素材不含外部私人文本或数据；合成原始数值在data/manifest.json标明CC0-1.0。
