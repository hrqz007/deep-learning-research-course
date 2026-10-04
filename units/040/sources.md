# 040 一手来源与查阅边界

所有教材推导、图和合成实验独立制作；没有复制论文图、商业教材或真实用户数据。查阅日期2026-10-04。

- S1：Cawley, G. C. 与 Talbot, N. L. C. (2010). On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. JMLR 11:2079–2107。实际打开全文，核对引言、§4/4.1模型选择噪声和§5.3全数据先调参再拆分的问题。https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf 。用于区分选择与评价；本讲独立Bernoulli/Gaussian机制例不冒充原论文实验。
- S2：Bergstra, J. 与 Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. JMLR 13:281–305。实际打开全文，核对PDF第3–4页有效维度讨论及§4。https://jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf 。本讲使用小网格展示完整单因素比较，未重跑论文神经网络基准，也未验证随机搜索普遍优越。
- S3：Bengio, Y. (2012). Practical Recommendations for Gradient-Based Training of Deep Architectures, arXiv:1206.5533v2。实际阅读作者HTML全文中的§3超参数范围和log尺度、§4.1梯度核验与受控拟合。https://arxiv.org/html/1206.5533v2 。用作调试问题的背景；具体阈值、合成数据与反例由本讲独立设定，不将经验建议写成定理。
- S4：PyTorch 2.7 SGD官方文档。实际核对更新伪代码中weight_decay进入梯度的位置和参数语义。https://docs.pytorch.org/docs/2.7/generated/torch.optim.SGD.html 。本讲以autograd和显式同步SGD更新实现权重专用L2，没有动量；不把这里的等价关系外推到Adam或任意优化器。
- S5：Python 3.12 gzip官方文档，GzipFile的filename/fileobj/mtime以及文本读取条目。https://docs.python.org/3.12/library/gzip.html 。采用空内部filename和mtime=0；实际记录环境内核对压缩与解压字节，不宣称所有压缩器版本逐位相同。

Bernoulli选择公式、集成平方误差恒等式、所有手算和逐样本梯度在本讲完整推导，可独立检查。配对bootstrap回顾022的方法，明确条件于固定训练与选择，不声称有限样本精确覆盖。技术版本与实际检查分开记录；参考文献的公开可读性不意味着给原作品添加新的版权许可。
