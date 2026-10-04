# 一手来源与阅读范围

本讲叙事、公式推导、纸笔数值、合成实验和15幅图均独立编写。以下来源用于检查方法边界，不把小型实验结果归给来源作者，也不复制其图表或长段原文。

1. Gavin C. Cawley 与 Nicola L. C. Talbot，2010，On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation，JMLR 11:2079–2107。[期刊原文页](https://jmlr.org/papers/v11/cawley10a.html)。核对：模型选择准则也有方差，反复优化验证结果可能产生选择偏差。本讲仍限定候选并在测试前冻结选择，不把有限网格理解成消除了全部选择不确定性。
2. scikit-learn官方文档，Cross-validation iterators for grouped data。[分组拆分说明](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)。核对：预测未见组时，训练与验证不能共享组；组数和行数可能不同。本讲自行生成实体拆分，没有调用scikit-learn，不要求安装该库。
3. PyTorch 2.7官方文档，Gradcheck mechanics。[梯度检查机制](https://docs.pytorch.org/docs/2.7/notes/gradcheck.html)。核对：反向自动微分的梯度可以与数值扰动检查，默认讨论实值双精度输入；本讲另写中心差分和NumPy梯度，不声称调用gradcheck即可证明全部输入正确。
4. PyTorch 2.7官方文档，Reproducibility。[可重复性边界](https://docs.pytorch.org/docs/2.7/notes/randomness.html)。核对：固定随机来源和确定性算法不保证跨版本、CPU/GPU和平台逐位相同。本讲保存具体环境与固定数据，不声称跨平台保证。
5. NumPy 2.3官方文档，numpy.linalg.lstsq。[最小二乘接口](https://numpy.org/doc/2.3/reference/generated/numpy.linalg.lstsq.html)。核对：求解最小二乘问题及返回系数、残差、秩和奇异值的语义。本讲用增广系统实现带权ridge，不显式求逆。

核查日期2026-10-04。正文关于整实体bootstrap的步骤直接对独立实体损失差构造，并明确其条件性近似范围；不把普通行重采样误称适合相关重复行。身份反例的方差公式从独立高斯生成过程直接推导。
