# 来源与查阅范围

查阅日期：2026-10-04。仅引用一手作者材料与官方文档。使用与实际运行一致的PyTorch2.7文档；stable入口发生重定向，故公开记录保留已实际打开的版本化链接。正文、手算、合成实验、图形和故障报告均独立撰写。未复制教材章节、论文图表或商业资料。

## S1 PyTorch：Gradcheck mechanics

https://docs.pytorch.org/docs/2.7/notes/gradcheck.html

实际阅读Notations与Default backward mode/Real-to-real部分，以及Fast gradcheck的实数方向核验说明。核对中央差分、逐输入坐标、反向雅可比及方向投影含义。本讲只实现实数标量目标的一阶核验，不声称实现复数、高阶或稀疏张量机制。正文误差阶、手算及盲方向反例独立推导。

## S2 PyTorch：torch.autograd.gradcheck.gradcheck

https://docs.pytorch.org/docs/2.7/generated/torch.autograd.gradcheck.gradcheck.html

实际阅读API参数、返回值、double precision提示、不可导点和重叠内存警告。代码显式使用float64，传入eps、atol、rtol；并用raise_exception=False记录预期失败。未沿用默认容差作为任何场景的通用保证。

## S3 PyTorch：Autograd mechanics

https://docs.pytorch.org/docs/2.7/notes/autograd.html

实际阅读图历史、非光滑函数梯度约定、requires_grad、grad modes、evaluation mode相关正文。用于区分路径中断、局部零导数以及eval与no_grad。ReLU与detach的具体数值均由本讲实际运行产生。

## S4 PyTorch：Optimizer.zero_grad

https://docs.pytorch.org/docs/2.7/generated/torch.optim.Optimizer.zero_grad.html

实际阅读set_to_none=True语义及None与零梯度行为差异。明确优化器只清空自己管理的参数；漏参探针改用model.zero_grad以清空所有模型参数。不同优化器的历史状态不能用普通SGD的一步等式替代。

## S5 PyTorch：Reproducibility

https://docs.pytorch.org/docs/2.7/notes/randomness.html

实际阅读开头跨版本/平台限制和Controlling sources of randomness。使用局部CPU Generator固定初始化并测试不修改全局随机状态；固定Python随机标签已写入CSV。未实施或测试CUDA、DataLoader多进程等后续章节方案。

## S6 Zhang, Bengio, Hardt, Recht, Vinyals (ICLR 2017)

Understanding deep learning requires rethinking generalization

https://arxiv.org/abs/1611.03530

实际阅读官方arXiv摘要、作者、版本和发表信息，核对其报告随机标签拟合现象。未把摘要查阅写成全文复现，未重新运行论文实验。本讲的8点回归记忆例不是其图像分类结果的定量复现，也不由此推断所有网络都会拟合所有随机标签。

## 证据分工

API语义由官方资料与运行探针互相核对；全部固定数值见outputs/，全文公式对应同一13参数trace；训练与干预数字由同目录脚本重算。独立高精度、分数、故障保护与Notebook执行范围见verification.json。任何没有实测的环境都明确列为限制。
