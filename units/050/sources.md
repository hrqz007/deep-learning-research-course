# 技术来源与证据对应

核验日期 2026-10-05。仅使用原始研究与官方版本文档支持外部技术事实。课程中的推导、数值例、代码、合成数据、图与实验结论均独立制作，不以外部来源代替运行证据。

1. Pascanu, R., Mikolov, T., Bengio, Y. (2013). On the difficulty of training recurrent neural networks. PMLR 28(3), 1310–1318.
   - 官方出版页：https://proceedings.mlr.press/v28/pascanu13.html
   - 论文：https://proceedings.mlr.press/v28/pascanu13.pdf
   - 支持：BPTT 的时间贡献、Jacobian 连乘及梯度消失/爆炸讨论，梯度范数裁剪的研究动机。
   - 限制：原文部分理论采用与本讲不同的状态参数化；不能照抄矩阵顺序。本讲采用 h=tanh(Ux+Wh前+b) 的列向量约定并自行推导。论文实验并非本讲的延迟符号实验。
2. PyTorch 2.7 官方 torch.nn.utils.clip_grad_norm_ 文档。
   - https://docs.pytorch.org/docs/2.7/generated/torch.nn.utils.clip_grad_norm_.html
   - 支持：拼接所有参数梯度意义下的范数、原地修改、返回裁剪前范数及非有限值错误选项。
   - 实际运行：torch 2.7.1+cpu；实际源码中分母的 1e-6 稳定项已在独立生产一步参考中纳入。
3. PyTorch 2.7 官方 Tensor.detach 文档。
   - https://docs.pytorch.org/docs/2.7/generated/torch.Tensor.detach.html
   - 支持：切断当前计算图、返回结果不要求梯度、共享底层存储。本文不对 detach 结果做破坏性原地更改。
4. PyTorch v2.7.1 官方 RNN 源码。
   - https://github.com/pytorch/pytorch/blob/v2.7.1/torch/nn/modules/rnn.py
   - 支持：RNN 非线性递推、输入与隐藏形状，以及输入/循环两项 bias 接口。相应 2.7 RNN 文档页面的检索未成功，改查可访问的官方固定版本源码，未把失败页面当作已核验来源。

## 原始科学证据

- protocol.json：首次训练前固定配置；outputs/results.json 保存其 SHA256。
- data/manifest.json：所有原始数组形状与 SHA256；generate_data.py 公开生成规则。
- outputs/hand_calculation.json：同一实例的全前向、反向、共享贡献、有限差分、同步更新与新前向。
- outputs/results.json：27 组全部逐步日志和终点指标，保留所有失败或较差的结果。
- outputs/trained_parameters.npz：全部终点模型参数。
- outputs/counterfactual.json：明确标注的事后诊断，未用于选择模型。
- outputs/environment.json：真实运行环境及 wall time。

稳定归约、padding 和状态重置的具体实现由独立数值/行为检查支持；检查通过不能证明全部输入域的正确性，也不能代替独立审阅。
