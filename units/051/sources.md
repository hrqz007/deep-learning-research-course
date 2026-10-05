# 一手来源与引用范围

所有数据、图、代码与数值账本为原创。来源用于框架定义与研究背景，不把原论文成绩当作本课结果。

- Cho et al., Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation, 2014：https://arxiv.org/abs/1406.1078 ，全文 https://arxiv.org/pdf/1406.1078 。原始门控/编码解码背景；本实现reset-after约定不声称论文权重兼容。
- Sutskever, Vinyals, Le, Sequence to Sequence Learning with Neural Networks, 2014：https://arxiv.org/abs/1409.3215 ，全文 https://arxiv.org/pdf/1409.3215 。条件序列生成背景，不复现其翻译任务。
- PyTorch2.7 GRUCell：https://docs.pytorch.org/docs/2.7/generated/torch.nn.GRUCell.html 。精确r/z/n定义、两份bias与参数shape；官方cell仅用于独立对照。
- PyTorch2.7 LSTM：https://docs.pytorch.org/docs/2.7/generated/torch.nn.LSTM.html 。门、c/h和投影变体的区分。本讲只验证无投影单cell解析例。

访问日期2026-10-05。主实验CPU float64、无注意力、无预训练、无外部语料，结论以data/outputs固定协议为准。确定性teacher-forced sequence exact与greedy exact等价的条件及证明为本讲推导，不是把不同评价指标混同。
