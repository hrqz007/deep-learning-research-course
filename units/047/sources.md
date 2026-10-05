# 一手来源与阅读边界

核对日期2026-10-05。教学数据、推导实例、图与代码独立制作，未复制论文图或大段文字。

1. Dosovitskiy等，An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale，作者论文v2：https://arxiv.org/html/2010.11929v2 。读取方法部分patch/位置/pre-norm与Hybrid架构，以及摘要中的预训练条件。没有下载权重或复现大规模性能。
2. Touvron等，Training data-efficient image transformers & distillation through attention，ICML2021，PMLR139:10347–10357：https://proceedings.mlr.press/v139/touvron21a.html 。读取论文摘要、出版信息与蒸馏训练条件；本实验未使用教师、预训练或DeiT训练配方。
3. PyTorch2.7 MultiheadAttention：https://docs.pytorch.org/docs/2.7/generated/torch.nn.MultiheadAttention.html 。读取接口和头shape；自写模块通过明确参数映射对照实际torch2.7.1+cpu。
4. PyTorch2.7 autograd：https://docs.pytorch.org/docs/2.7/autograd.html 。读取saved_tensors_hooks相关接口，用于CPU保存张量记录。
5. PyTorch2.7 max_memory_allocated：https://docs.pytorch.org/docs/2.7/generated/torch.cuda.max_memory_allocated.html 。读取峰值已分配显存定义，仅解释未来GPU补测口径；本机CUDA不可用，未测显存。

原课纲要求记录资源，本文提供真实CPU保存张量口径和解析容量，明确标注GPU峰值未测，未把估计当实测。未来使用其他优化后端需重新验证。
