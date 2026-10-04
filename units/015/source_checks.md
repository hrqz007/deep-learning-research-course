# 第015单元来源核查

实际访问日期：2026-10-04。全部使用作者公开材料、开放作者教材或官方软件文档。正文组织、推导、例子、图、数据、代码和练习独立编写，未复制来源图表或大段正文。

1. Goodfellow、Bengio、Courville：[Deep Learning Chapter 4 Numerical Computation](https://www.deeplearningbook.org/contents/numerical.html)。实际读取4.3中Hessian定义、连续混合偏导、方向二阶导数、二阶模型以及临界点判别相关部分。核对公式4.6至4.10、页85至89附近内容。本文不采用该章对一般非正规矩阵条件数的简略特征值比说法；κ=λ最大/λ最小只用于本讲正定H。原文采用列向量，本文系统改为行向量并逐式检查形状。

2. Boyd、Vandenberghe：[Convex Optimization slides](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf)。实际读取PDF第53页的一阶条件、第54页二阶条件，以及第295至299页的梯度下降与二次函数分析、第303至305页尺度/度量相关部分。定义域为凸集的条件保留；正文对固定二维二次目标另作完整独立证明。此前尝试访问同站教材PDF bv_cvxbook.pdf返回工具内部错误，因此不把未读取的教材正文列为已验证来源，改查实际可读的作者讲义。

3. OpenStax作者教材：[Calculus Volume 3 4.7 Maxima/Minima Problems](https://openstax.org/books/calculus-volume-3/pages/4-7-maxima-minima-problems)。实际读取局部/绝对极值、临界点、二阶导数判别与判别不充分情况。二维一般Hessian论证仍以明确C²假设、特征方向及余项为本讲推导，不把标量判别式当作高维通用方法。

4. NumPy 2.3：[numpy.linalg.eigh](https://numpy.org/doc/2.3/reference/generated/numpy.linalg.eigh.html)。实际读取实对称/复Hermitian适用对象、特征值按升序返回、特征向量按列返回与LinAlgError条件。本实验只接受实数精确对称的2×2矩阵，不自动“修复”输入。

Taylor积分余项、99组有理数核验、287个递推状态、特定步长反例与缩放实验均是本单元独立展开。有限CPU计算是机制检验，不能证明一般神经网络的收敛，也不是实际训练速度或泛化效果的实测。
