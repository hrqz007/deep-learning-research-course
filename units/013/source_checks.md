# 来源核验

核验日期：2026-10-04。仅使用官方文档、作者公开讲授材料和原始论文。全部教学叙述、图、数值与测试独立制作，不复制来源图像或长段正文。公开可读不代表重新许可来源原文；此文件只记录引用与适用边界。

## 1 MIT 18.06 Lecture 29 Singular value decomposition
[原始来源](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-29-singular-value-decomposition/)

课程页确认任意矩阵SVD以及正交、对角、正交三个因子；正文引用一般存在性，不宣称本讲证明谱定理。

## 2 LAPACK Users Guide Singular Value Decomposition
[原始来源](https://netlib.org/lapack/lug/node53.html)

实矩形矩阵的SVD定义、非负降序奇异值、左右向量及直接数值算法路线。

## 3 NumPy 2.3 numpy.linalg.svd
[原始来源](https://numpy.org/doc/2.3/reference/generated/numpy.linalg.svd.html)

full_matrices=False的返回形状、s为降序一维数组、Vh已转置、重构与_gesdd接口；本讲限定实二维输入。

## 4 MIT 18.065 Lecture 7 Eckart Young
[原始来源](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/resources/lecture-7-eckart-young-the-closest-rank-k-matrix-to-a/)

官方课程页及逐字稿第1–4页：截断SVD在谱与Frobenius等指定范数下给最佳低秩近似。一般谱范数最优性引用，Frobenius部分在本讲独立写出投影证明。

另直接检查[官方逐字稿](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/70996912a170cf9c6ebb018f03c1fc85_Y4f7K9XF04k.pdf)相关页。

## 5 NumPy numpy.linalg.cond
[原始来源](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html)

p=None使用SVD的2范数条件数；条件数依赖所用范数；本讲推导仅针对可逆方阵右端扰动。稳定入口检查时页面标为v2.5，实际执行为NumPy2.3.5，不混称版本。

## 6 NumPy 2.3 numpy.linalg.solve
[原始来源](https://numpy.org/doc/2.3/reference/generated/numpy.linalg.solve.html)

解方阵满秩线性系统；行方程需转置；奇异情况抛LinAlgError。

## 7 NumPy 2.3 numpy.linalg.matrix_rank
[原始来源](https://numpy.org/doc/2.3/reference/generated/numpy.linalg.matrix_rank.html)

以超过tol的奇异值数计数；默认阈值考虑尺寸与机器精度；官方说明测量噪声可能需要别的阈值。

## 8 Hu et al 2021 LoRA
[原始来源](https://arxiv.org/abs/2106.09685)

原论文第4.1节固定W0并学习低秩乘积增量。正文把列约定改写为已教的行约定，因子命名为AB；只借用参数化，不引用模型效果数字或声称复現。

另直接检查[原论文全文](https://arxiv.org/html/2106.09685v2)第4.1节。

## 证明边界

一般SVD存在性依赖实对称谱定理，正文明确引用。截断Frobenius误差、算子范数与最大奇异值、Frobenius最优性、可逆方阵右端扰动的条件数界分别给出推导。谱范数下最佳低秩近似的一般最优性仅引用，不冒充本讲已证明。数据与权重压缩、任务指标与矩阵误差、低秩增量与完整权重被明确区分。
