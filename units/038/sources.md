# 038 来源与实际使用范围

2026-10-04核对。这里列出原始论文、出版方记录和官方接口。讲义中的例子、数值、图、证明的教学展开和实验均独立制作；没有打包第三方论文全文或复制论文图。公开链接不改变原作者版权。

1. Belkin, Hsu, Ma, Mandal. Reconciling modern machine learning practice and the bias-variance trade-off. PNAS, 2019. [作者公开版本1812.11118v2](https://arxiv.org/abs/1812.11118v2)，[全文](https://arxiv.org/html/1812.11118v2)。检查插值阈值、双下降描述与讨论部分对优化、正则化和采样宽度的限制。本讲未复现论文真实数据实验，仅构造可解释的线性合成例子。

2. Hastie, Montanari, Rosset, Tibshirani. Surprises in High-Dimensional Ridgeless Least Squares Interpolation. [作者公开版本1903.08560v5](https://arxiv.org/abs/1903.08560v5)，[全文](https://arxiv.org/html/1903.08560v5)。检查风险定义、零初始化GD的行空间证明、各向同性比例极限、遗漏特征与Ridge部分。正文只给最简单情形的有限谱分解和条件化说明，不声称概括全部68页理论或证明随机矩阵极限。

3. Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals, and Systems 2, 303–314, 1989. [出版方](https://doi.org/10.1007/BF02551274)，[原论文PDF学术镜像](https://papers.baulab.info/papers/Cybenko-1989.pdf)。核对单位立方体、连续sigmoid、连续目标的一致逼近以及第3节定理2。没有把该版本改称ReLU定理，没有转述为训练算法或样本效率保证。

4. Bartlett, Harvey, Liaw, Mehrabian. Nearly-tight VC-dimension and Pseudodimension Bounds for Piecewise Linear Neural Networks. JMLR 20(63), 1–17, 2019. [原始期刊页面](https://www.jmlr.org/papers/v20/17-612.html)，[作者版本](https://arxiv.org/abs/1703.02930)。核对摘要对权重数、层数、非线性单元及激活类型的区别。本讲不复述或证明其复杂度上下界，只据此提醒参数数目不等于完整容量概念。

5. Hoeffding. Probability Inequalities for Sums of Bounded Random Variables. JASA 58(301), 13–30, 1963. [出版方记录及摘要](https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500830)。核对原始出处和独立有界变量范围；全文访问未完成。正文将标准Hoeffding双侧不等式作为明确命名的概率工具，完整推导有限候选的联合界，没有声称阅读原论文完整证明。

6. NumPy2.3官方接口：[svd](https://numpy.org/doc/2.3/reference/generated/numpy.linalg.svd.html)、[lstsq](https://numpy.org/doc/2.3/reference/generated/numpy.linalg.lstsq.html)。检查紧致SVD、最小二乘最小范数解、rcond和数值rank。我们的风险定义、SVD滤波公式与异常守卫由本讲自己实现。

7. PyTorch2.7官方入口：[autograd.grad](https://docs.pytorch.org/docs/2.7/generated/torch.autograd.grad.html)。此次网页提取不可用，实际检查已安装2.7.1+cpu中torch.autograd.grad的docstring与源代码，并执行逐样本参数导数验证。没有把另一个版本的网页描述当成本地实现。版本安装参考[官方历史版本](https://pytorch.org/get-started/previous-versions/#v271)。

正文的有限类界不能直接套Gaussian平方损失实验。连续参数模型、无界损失、相关数据或数据依赖候选需要另行处理。比例渐近曲线不是24样本有限实验的精确预报。资料引用用于限定条件，不替代可重跑证据。
