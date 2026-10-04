# 第027单元来源与原创边界

核查日期：2026-10-04。正文的批次指标推导、两层手算、5条数据、26参数夹具、八幅图和NumPy实现均独立编写；下列一手来源用于核对接口定义与方法边界。未复制商业教材或他人的图。

1. NumPy官方，[Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)：实际读取General broadcasting rules和二维/多维广播示例，核对尾轴对齐和长度为1的扩展。
2. NumPy官方，[numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)：实际读取Parameters、Notes，核对二维矩阵乘法签名与@。
3. NumPy官方，[numpy.sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)：实际读取axis、keepdims、Notes，核对求和轴与浮点累加限制。
4. Stanford CS231n原课程，[Backpropagation](https://cs231n.github.io/optimization-2/)：实际读取链式法则、局部计算和向量化梯度说明，用来核对本讲标量到矩阵的推导方向。
5. Stanford CS231n原课程，[Gradient Checks](https://cs231n.github.io/neural-networks-3/)：实际读取Gradient Checks段，包括中心差分、相对误差、浮点精度、折点、步长和单点检查的局限。本文双容差门槛为本夹具自定，并非声称课程规定的统一标准。
6. PyTorch官方，[CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)：实际读取logits输入、class-index和probability-target两种公式、weight/ignore_index及reduction。硬标签加权mean按有效权重和归一，不能把所有mean都理解成简单除batch大小。
7. PyTorch官方，[Numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)：实际读取Batched computations or slice computations及浮点累加说明，只引用不保证逐位相同的范围提醒。文档个别示例不作为本讲计算依据。

stable页面访问时NumPy显示2.5，PyTorch跳转2.14；这不代表实际执行了这些版本。本地核心实验为NumPy2.3.5；没有运行PyTorch、GPU或付费API。7个页面均实际读取相关正文，而非仅搜索摘要。来源状态可机读版本见source-checks.json。
