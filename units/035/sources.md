# 第035讲来源查阅记录

访问日期：2026-10-04。以下为实际打开并检查的一手资料；正文的数值例子、推导展开、实验与图均独立制作，没有复制论文配图。公式来自共同的数学知识与明确推导，不以综述或商业教材替代一手依据。

|编号|来源和实际查阅位置|用于何处|限制|
|---|---|---|---|
|S1|Glorot, X. & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. AISTATS, PMLR 9:249–256. PDF第5页（印刷页253），§4.2.1，公式2–16；另看摘要和§3的激活讨论。https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf |核对线性区假设、fan_in/fan_out矛盾、2/(fan_in+fan_out)折中。|原文基准不是本单元数据；未重现原论文的大型训练结论。|
|S2|He, K., Zhang, X., Ren, S. & Sun, J. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. arXiv:1502.01852. PDF第3–5页，§2.2，公式5–15。https://arxiv.org/pdf/1502.01852 |核对ReLU非零均值、二阶矩与方差差别、前向/反向独立假设及leaky/PReLU系数。|先尝试CVF官方PDF访问时读取失败，随后实际查阅作者arXiv全文。没有将原文特定架构的实验推广成任意架构定理。|
|S3|PyTorch 2.7 torch.nn.init，xavier_uniform_、xavier_normal_、kaiming_uniform_、kaiming_normal_及Notes。https://docs.pytorch.org/docs/2.7/nn.init.html |核对标准差、gain、mode和W[fan_out,fan_in]布局；测试当前2.7.1+cpu与公式一致。|文档页面为2.7系列，实际执行版本记录为2.7.1+cpu；没有声称测试所有算子与设备。|
|S4|PyTorch 2.7 Autograd mechanics，“Gradients for non-differentiable functions”。https://docs.pytorch.org/docs/2.7/notes/autograd.html |区分ReLU零点不可导与框架的边界选择；同时通过实际PyTorch反向测试0处返回0。|单独ReLU API页面最初读取失败；本讲边界行为以已读取autograd说明和实际算子测试双重支持。|

## 本单元的独立推导边界

新层权重独立、零均值并与前层表示独立时，条件期望展开可删除权重交叉项，不必额外要求输入坐标独立。固定权重对数据取方差时，则必须保留输入协方差。这个区分在正文中逐步展开，并用精确有限枚举和相关输入反例测试，不假称为某篇论文的原句。

反向将门、权重和上游梯度拆分是额外近似；本单元用简单有限概率反例说明乘积期望不能无条件分解。Jacobian的平均平方奇异值不能保证每条方向保持，是两个2×2矩阵的直接计算。

实验比较只对应本单元发布的CSV、网络、48配置和固定SGD学习率。未查阅或依赖二手博客作关键公式依据；未声称大模型、ImageNet、GPU、fresh Anaconda或浏览器传输已复现。

## 压缩文件的读取与可重现封装

S5. Python 3.12官方gzip文档（实际查阅GzipFile、open、compress、decompress条目）：https://docs.python.org/3.12/library/gzip.html 。用于核对文本模式读取、内部filename、mtime和压缩级别参数。文档说明gzip使用zlib；本讲以GzipFile固定空filename和mtime=0，并以实际重跑验证当前环境的压缩字节一致，不外推到所有压缩器版本。
