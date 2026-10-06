# 原始来源与适用边界

核对日期：2026-10-06。正文、手算、代码和图为课程原创；下列资料用于核验算法背景，不是本次测量数据。

1. Holtzman, A., et al. The Curious Case of Neural Text Degeneration. ICLR 2020. https://arxiv.org/abs/1904.09751 。原作者论文，支持nucleus采样及解码策略会影响生成退化的背景。本课只运行短合成符号句，未复现论文的大规模语言实验。
2. Hugging Face. How caching works. https://huggingface.co/docs/transformers/cache_explanation 。官方文档，核对逐层K/V、因果历史不变和推理缓存的概念。本课自行实现，不依赖该库，也未照搬其API。
3. PyTorch. torch.multinomial. https://docs.pytorch.org/docs/2.7/generated/torch.multinomial.html 。官方版本化文档，核对非负有限权重、非零总和、generator参数。稳定版入口本次跳转异常，改用可核验版本页；本次实际运行软件版本以outputs/results.json的runtime为准。
4. 本课程055公开教学检查点，来源路径、提交和内容摘要见data/provenance.json。实验固定复用该检查点，不将055的测试集用于选择本讲解码参数。

所有本课结果可从outputs/results.json及data/原文件复算。缓存计时是当前CPU单线程微基准，不是外部来源支持的通用性能承诺。
