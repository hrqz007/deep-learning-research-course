# 一手来源核验：009

核验日期：2026-10-04。通过公开网页读取官方参考和作者教材；没有复制教材段落或外部图片。正文和实验的数字、图、循环、反例与讲解独立编写。版本锁定NumPy 2.3系列，运行实测2.3.5。

| 来源与已核对内容 | 用途与边界 |
|---|---|
| [NumPy matmul](https://numpy.org/doc/2.3/reference/generated/numpy.matmul.html)，二维(n,k),(k,m)→(n,m)、不接受标量、1D特殊提升规则、@语义 | 支持算子契约；本讲严格函数主动只接受2D，未覆盖高维矩阵堆叠 |
| [NumPy transpose](https://numpy.org/doc/2.3/reference/generated/numpy.transpose.html)，二维轴交换，一维返回形状不变的视图 | 支持一维.T不能增加行列轴；数学“转置不等于逆”由原创反例证明 |
| [NumPy Broadcasting](https://numpy.org/doc/2.3/user/basics.broadcasting.html)，从右向左对齐，相等或1兼容，不必真的复制 | 支持共享偏置的实现；不据此作本例速度或峰值内存结论 |
| [NumPy broadcast_to](https://numpy.org/doc/2.3/reference/generated/numpy.broadcast_to.html)，只读视图、多元素可能指向同一存储位置 | 支持广播视图解释，配套代码实际检查shares_memory、writeable与strides |
| [NumPy ndarray.nbytes](https://numpy.org/doc/2.3/reference/generated/numpy.ndarray.nbytes.html)，元素字节总量，不含数组对象的非元素属性 | 结合广播视图实测说明逻辑字节数不等于新增分配量 |
| [Margalit与Rabinoff，Interactive Linear Algebra §3.3](https://textbooks.math.gatech.edu/ila/linear-transformations.html)，线性定义、原点必要条件、标准基与矩阵列关系 | 作者原教材为列向量约定，本讲完整转为行样本；网页抽取对部分公式显示不全，因此本讲所有公式另由逐项代数与独立分数程序验证，不依赖抽取丢失公式 |

矩阵结合律、转置乘积式、行基像、仿射组合与中点性质均在正文给出逐项或分配律推导；不把软件测试当作普遍数学证明。正文合成数据和固定规则没有外部任务性能来源，也不需要此类来源。无训练或真实设备性能主张。
