# 033 来源与核查范围

查阅日期：2026-10-04。只使用以下一手论文、官方文档和实际安装版本源码核对概念及API行为；讲义文字、手算、合成数据和图为课程独立制作，未复制论文图表或商业教材内容。公开可读不等于为外部材料附加版权许可。

1. [PyTorch 2.7 SGD](https://docs.pytorch.org/docs/2.7/generated/torch.optim.SGD.html)。实际阅读算法与Notes：经典动量、首个缓冲的初始化、与把LR放进速度递推的差别。本讲限定无dampening、无Nesterov、无权重衰减；未外推到所有SGD变体。
2. [PyTorch 2.7 torch.optim](https://docs.pytorch.org/docs/2.7/optim.html)。实际阅读How to adjust learning rate附近内容，核对scheduler.step与optimizer.step次序。单个调度器的时间单位仍需由调用位置决定。
3. [PyTorch 2.7 StepLR](https://docs.pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.StepLR.html)。核对示例在每epoch末调用及step_size单位。首次直接docs域访问失败，随后从pytorch.org版本链接重定向成功读取，不能把首次失败写成已读。
4. [PyTorch 2.7 CosineAnnealingLR](https://docs.pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)。核对余弦形状、T_max以及没有重启。课程用自定义离散端点的LambdaLR，不声称逐点等同于默认CosineAnnealingLR。
5. [PyTorch 2.7 LambdaLR](https://docs.pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.LambdaLR.html)与[v2.7.1官方源码](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/optim/lr_scheduler.py)。首次docs域页面不可达，后经pytorch.org版本地址读取成功；另实际读取本地2.7.1+cpu的构造、state_dict和load_state_dict实现，核对普通lambda函数本身不被保存及初始化计数行为。源码摘要见source-checks.json，未把源码全文打包发布。
6. Sutskever, Martens, Dahl, Hinton. [On the importance of initialization and momentum in deep learning](https://proceedings.mlr.press/v28/sutskever13.html), ICML 2013。实际阅读论文第2节（PDF第2页附近）区分经典动量和Nesterov。这里只引用机制背景，未复现论文的深层或循环网络实验。
7. Goyal等. [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677), 2017。实际查阅PDF第2.2节（第3页）渐进预热和第5节预热对照（第7至8页）。用于说明预热有具体实验适用条件；未挪用其ImageNet结果作为本课程小网络成绩。
8. Loshchilov, Hutter. [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983), ICLR 2017。实际查阅官方arXiv摘要与版本记录，用于识别余弦衰减的重启研究背景。没有宣称逐段查阅全文或实现其重启实验。

本讲的稳定性区间、EMA权重和方差、两次9参数手算均在正文独立推导；Fraction、NumPy、PyTorch和有限差分各承担不同数值核验角色。工具显示网页可达不等于整篇都已阅读，以上明确记载了实际查阅范围。
