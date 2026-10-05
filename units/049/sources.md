# 一手来源与引用范围

所有图、数据、讲义推导和代码均为本课原创。2026-10-05访问下列一手来源，不使用第三方解读替代方法定义。

1. Unicode Consortium，UAX #29，Unicode Text Segmentation：https://www.unicode.org/reports/tr29/ 。用于字素、码点与文本分段背景。代码没有实现完整UAX29分段器。
2. Unicode Consortium，UAX #15，Unicode Normalization Forms：https://unicode.org/reports/tr15/ 。用于规范化背景；本实现明确不规范化，精确保留UTF-8字节。
3. Sennrich, Haddow, Birch，Neural Machine Translation of Rare Words with Subword Units，ACL2016：https://aclanthology.org/P16-1162/ ，原文PDF https://aclanthology.org/P16-1162.pdf 。用于子词/BPE研究背景。本文自定义的是字节级、允许跨空格但不跨文档的教学算法，不声称论文实现兼容。
4. Kudo, Richardson，SentencePiece，EMNLP2018：https://aclanthology.org/D18-2012/ ，原文PDF https://aclanthology.org/D18-2012.pdf 。用于了解独立tokenization系统；本实验没有依赖或复现SentencePiece。
5. PyTorch2.7官方Embedding文档：https://docs.pytorch.org/docs/2.7/generated/torch.nn.Embedding.html 。用于shape、padding_idx和梯度选项的语义；本文采用dense梯度SGD，不启用max_norm、sparse、scale_grad_by_freq。

论文或文档中的实验成绩没有复制为本课结果。本课所有数值以data、outputs与固定协议为准。来源页面发生版本更新不等于本课软件环境自动更新。
