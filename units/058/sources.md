# 原始来源与适用边界

核验日期：2026-10-06。只使用作者教材、期刊原论文和官方软件文档。以下链接支持概念与API说明；本单元的数据、公式算例、实验结果和原创图均由随包代码独立构造，没有搬用论文图表。

1. Goodfellow, Bengio, Courville, Deep Learning, Chapter 14 Autoencoders，2016。https://www.deeplearningbook.org/contents/autoencoders.html 。用于核对编码/解码、欠完备、线性平方损失与PCA、去噪与活动约束的概念。这里只作简短概念指引；本课不复刻该书的完整理论讨论。
2. Vincent, Larochelle, Lajoie, Bengio, Manzagol，Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion，JMLR 11(110):3371–3408，2010。https://www.jmlr.org/papers/v11/vincent10a.html 。核对去噪输入恢复干净目标的研究来源。本课不是该论文图像分类或逐层堆叠实验的复现。
3. PyTorch官方 MSELoss 文档。https://docs.pytorch.org/docs/2.14/generated/torch.nn.MSELoss.html 。核对mean会对所有元素平均，而不是仅对batch平均。引用版本2.14与本次执行的2.14.1+cpu对应。
4. NumPy官方 numpy.linalg.svd 文档。https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html 。核对U、s、Vh顺序、full_matrices=False形状和奇异值递减顺序。网页当前版本可能晚于本实验NumPy2.3.5；本课使用的基础SVD接口由测试实际验证。
5. PyTorch官方 Reproducibility 文档。https://docs.pytorch.org/docs/2.14/notes/randomness.html 。核对随机源、确定性算法及跨平台/版本复现限制。固定种子不是跨硬件逐位一致承诺。

本课报告的经验结论仅来自outputs/results.json；每行指标均可由保存的输入、表示、权重与探测器独立复算。外部来源不为本合成实验的48%准确率背书。
