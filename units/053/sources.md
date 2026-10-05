# DL053 一手来源

核验日期2026-10-05。引用来源用于概念/API核对，数值、数据与图均为本单元原创。未转载论文图片或正文。

1. Vaswani et al. (2017), Attention Is All You Need. https://arxiv.org/abs/1706.03762 。核对多头、FFN、encoder-decoder和正弦位置编码。
2. Xiong et al. (2020), On Layer Normalization in the Transformer Architecture. https://arxiv.org/abs/2002.04745 。核对pre/post与梯度路径动机，不声称普遍最优。
3. Su et al. (2021), RoFormer. https://arxiv.org/abs/2104.09864 。核对旋转位置与相对内积；不把论文结论当本实验结果。
4. PyTorch 2.7 TransformerEncoderLayer. https://docs.pytorch.org/docs/2.7/generated/torch.nn.TransformerEncoderLayer.html 。核对norm_first、batch_first、eps、mask。
5. PyTorch 2.7 LayerNorm. https://docs.pytorch.org/docs/2.7/generated/torch.nn.LayerNorm.html 。核对D分母方差和归一化轴。
