# 一手来源与实际阅读范围

检查日期：2026-10-04。正文与图独立撰写；未复制论文插图、商业教材或模型权重。下列链接用于核对概念与API。没有把浏览全文目录或摘要写成阅读全文。

S1. Yosinski, Clune, Bengio, Lipson. How transferable are features in deep neural networks? NeurIPS 2014.
https://proceedings.neurips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf
实际读取官方9页PDF的摘要、第1节及第2节（PDF第1至3页，抽取文本约第0至136行），核对源/目标划分、冻结/微调、专用性与共同适应两种限制。未阅读全文的实验第4节及附录，不声称重现论文数值。正文只用其研究问题与对照思想，本单元五参数合成实验为原创。

S2. PyTorch 2.7 Autograd mechanics.
https://docs.pytorch.org/docs/2.7/notes/autograd.html
实际读取How autograd encodes the history，以及Locally disabling gradient computation下的Setting requires_grad、Grad Modes、No-grad、Inference Mode、Evaluation Mode段落（约第30至118行）。未通读复数、多线程及全部后续章节。核对叶参数梯度、输入需要梯度时的路径保留、eval与梯度控制的正交关系。并用本地2.7.1+cpu真实代码测试，不假定stable网页当前版本与本地一致。

S3. PyTorch 2.7 torch.optim.
https://docs.pytorch.org/docs/2.7/optim.html
实际读取Constructing it、Per-parameter options、Taking an optimization step，以及确定性参数顺序提示（约第33至114行）。未逐项读取所有优化器API。核对参数组学习率与默认值覆盖语义；本单元实际使用无动量SGD，不引入Adam相关结论。

S4. PyTorch 2.7 torch.Tensor.detach.
https://docs.pytorch.org/docs/2.7/generated/torch.Tensor.detach.html
读取该短API正文全部内容（约第27至40行，不含导航），核对返回张量脱离当前图、不要求梯度，以及与原张量共享存储。没有把detach当复制或参数冻结的同义词。

公式来源说明：五参数链式法则、条件方差分解、岭回归正规方程和预算计算在本讲完整推导，属于本单元独立数学推演，不依赖来源中的未展示结论。所有图的数值源可从make_figures.py和outputs追溯。未引用额外未打开的搜索结果。
