# 034 一手来源与实际核对范围

访问并核对日期：2026-10-04。正文叙事、所有数字例子、图、代码、练习均独立编写。公式以本讲明确的符号和参数约定展开，不照抄文献文字或商业教材。

1. Kingma与Ba，Adam A Method for Stochastic Optimization，ICLR2015，[作者论文](https://arxiv.org/pdf/1412.6980)。实际阅读算法1、第2节更新、第3节零初始化偏差修正；核对逐坐标平方/根号、beta幂及分布非平稳时的限制。未将第4节或附录收敛论证作为已核验定理，也未复现论文基准实验。本讲的完整两步网络例子为独立计算。
2. Loshchilov与Hutter，Decoupled Weight Decay Regularization，ICLR2019，[作者论文](https://arxiv.org/pdf/1711.05101)。实际检查第2节、算法2、命题1至3的适用对象及附录A的代数证明，特别区分固定预条件矩阵的解释与真实Adam变化状态。没有复现图像基准、Bayesian解释或文中大规模结论；框架中的lambda参数化在正文单独写明。
3. [PyTorch2.7 Adam官方文档](https://docs.pytorch.org/docs/2.7/generated/torch.optim.Adam.html)。核对算法、betas、根号外eps、weight_decay、decoupled_weight_decay选项以及foreach/fused接口。本讲基础Adam使用默认耦合语义；实际数值版本为2.7.1+cpu，不是泛指任意版本。
4. [PyTorch2.7 AdamW官方文档](https://docs.pytorch.org/docs/2.7/generated/torch.optim.AdamW.html)。核对旧参数衰减、数据梯度状态、参数组、state_dict及恢复时按参数组顺序匹配的说明。实际运行二维机制和状态恢复检查；未测试磁盘/跨版本或GPU实现。
5. [PyTorch2.7 SGD官方文档](https://docs.pytorch.org/docs/2.7/generated/torch.optim.SGD.html)。核对耦合weight_decay与动量、初次buffer语义。训练比较显式使用momentum=.8、默认dampening=0、无Nesterov，不把plain SGD的L2代数等价直接推广到不同动量实现。
6. [PyTorch2.7 Optimizer.zero_grad官方文档](https://docs.pytorch.org/docs/2.7/generated/torch.optim.Optimizer.zero_grad.html)。核对None与数值0的更新差异，且用真实AdamW在已有动量的标量例上运行。只作为该框架具体实现语义。
7. [PyTorch官方历史版本安装页](https://pytorch.org/get-started/previous-versions/#v271)。核对2.7.1 CPU安装入口；运行时使用已存在隔离CPU环境，没有声称本讲又完成一次Anaconda全新安装。

OpenReview上On the Convergence of Adam and Beyond的页面与PDF尝试返回浏览器验证页，因此未作为本讲已实际读到的来源或已证明结果。本文不依赖其证明，只明确不作一般非凸收敛保证。

来源用于算法与实现核对，不意味着来源作者支持本讲的合成实验结论。公开访问不改变原文的许可，也不授予对第三方材料的再许可。
