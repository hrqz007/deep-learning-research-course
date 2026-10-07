# 一手来源与使用边界

核验日期：2026-10-07。以下为实际打开核对的官方文档与作者论文。正文采用原创解释、算例和图，不复制来源图。版本固定为2.14文档，安装其他版本时应再次核对API。

1. PyTorch Type Info

https://docs.pytorch.org/docs/2.14/type_info.html

使用范围：finfo的eps与tiny含义。

2. PyTorch AMP

https://docs.pytorch.org/docs/2.14/amp.html

使用范围：autocast按设备与算子选dtype。

3. Mixed Precision Training

https://arxiv.org/abs/1710.03740

使用范围：混合精度与loss scaling研究背景。

4. PyTorch AMP examples

https://docs.pytorch.org/docs/2.14/notes/amp_examples.html

使用范围：累积时scale不变、unscale和裁剪时机。

5. Training Deep Nets with Sublinear Memory Cost

https://arxiv.org/abs/1604.06174

使用范围：用重算交换激活保存空间。

6. PyTorch checkpoint

https://docs.pytorch.org/docs/2.14/checkpoint.html

使用范围：use_reentrant与preserve_rng_state语义。

结果数字来自本单元outputs/results.json，不能由参考文档替代实测。
