# DL055 一手来源

核验日期2026-10-05。未转载论文图表或段落；合成文法、数字、全部图和示例为本单元原创。

1. Kaplan et al. Scaling Laws for Neural Language Models (2020). https://arxiv.org/abs/2001.08361 。经验规模关系及条件；不采用其指数为本实验结论。
2. Hoffmann et al. Training Compute-Optimal Large Language Models (2022). https://arxiv.org/abs/2203.15556 。参数、数据与计算配置；本课不做前沿外推。
3. PyTorch2.7 AdamW. https://docs.pytorch.org/docs/2.7/generated/torch.optim.AdamW.html 。偏差校正和解耦衰减。
4. PyTorch2.7 clip_grad_norm_. https://docs.pytorch.org/docs/2.7/generated/torch.nn.utils.clip_grad_norm_.html 。全局梯度范数和裁剪。
5. Python resource. https://docs.python.org/3/library/resource.html 。接口与平台依赖；KiB到字节转换仅用于实测Linux。
6. PyTorch2.7 max_memory_allocated. https://docs.pytorch.org/docs/2.7/generated/torch.cuda.max_memory_allocated.html 。仅解释GPU测量字段，未在本实验测GPU。
7. Vaswani et al. Attention Is All You Need (2017). https://arxiv.org/abs/1706.03762 。结构背景，实验模型配置以本课代码为准。
