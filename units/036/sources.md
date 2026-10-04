# 一手来源与实际阅读范围

以下链接均在制作时实际打开；另检查实际安装的PyTorch2.7.1+cpu代码或docstring。正文推导、数值表、图和实验均独立编写，不复制论文图。来源支持定义和接口语义，不替代本讲实验记录。

1. [Ioffe与Szegedy 2015 Batch Normalization](https://proceedings.mlr.press/v37/ioffe15.pdf)。阅读第2至3节、批归一化计算及批统计参与反向的说明。采用其批统计定义；未复现ImageNet或速度结果，未声称内部协变量偏移是已确立的唯一因果机制。
2. [Ba等2016 Layer Normalization](https://arxiv.org/pdf/1607.06450)。阅读第2至3节，尤其式3的层内统计与样本之间的区别。没有把其RNN结果当成本讲证据。
3. [PyTorch2.7 BatchNorm1d](https://docs.pytorch.org/docs/2.7/generated/torch.nn.BatchNorm1d.html)。完整阅读shape、训练前向biased variance、运行unbiased variance、momentum约定、track_running_stats=False与eval语义。
4. [PyTorch2.7 LayerNorm](https://docs.pytorch.org/docs/2.7/generated/torch.nn.LayerNorm.html)。阅读normalized_shape、尾部轴、elementwise_affine、方差与train/eval定义。
5. [PyTorch2.7 Module](https://docs.pytorch.org/docs/2.7/generated/torch.nn.Module.html#torch.nn.Module.eval)。阅读eval/train、buffer与state_dict相关定义；实际内存实验核对training布尔值不由state_dict恢复。
6. [PyTorch2.7 no_grad](https://docs.pytorch.org/docs/2.7/generated/torch.no_grad.html)。阅读梯度模式语义；用实际四组合实验区分模块模式。本讲未扩展成所有工厂函数或forward-mode AD行为的完整教程。
7. [PyTorch官方2.7.1历史安装命令](https://pytorch.org/get-started/previous-versions/#v271)。沿用既有实际CPU安装环境；没有对每个系统重新安装验证。

安装源码核对：torch.nn.modules.batchnorm._BatchNorm.forward中实际计数、momentum=None的1/num_batches_tracked、buffers为None时选择当前批统计的分支。代码以实际2.7.1 CPU执行为准，不把未来版本页面当作旧版本实测。

科学边界：epsilon、统计组、仿射参数和评价状态的机制可由本讲数值核对；归一化是否改善任务效果、如何比较部署校准、不同架构的选择仍需对应数据与实验。无测试时适应、分布式或大规模性能结论。
