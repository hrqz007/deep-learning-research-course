# 一手来源与用途

检查日期：2026-10-05。以下均为原作者论文或框架官方资料。课程的推导、数值账本、数据、代码和图为原创；没有复制论文图表或大段文本。

1. He, Zhang, Ren, Sun. Deep Residual Learning for Image Recognition. 2015/2016. https://arxiv.org/html/1512.03385v1 。核对§3残差映射、逐元素相加、尺寸改变时投影及普通网络对照；不把该论文的大规模数据结果归于本讲小实验。
2. He, Zhang, Ren, Sun. Identity Mappings in Deep Residual Networks. 2016, v3. https://arxiv.org/html/1603.05027v3 。核对恒等捷径、加法后激活的条件与预激活动机。本讲没有声称完整复现其结构或成绩。
3. PyTorch v2.7.1官方conv.py。https://github.com/pytorch/pytorch/blob/v2.7.1/torch/nn/modules/conv.py 。核对Conv2d互相关语义、NCHW/OIHW形状、stride/padding、参数定义；数值实验实际使用torch 2.7.1+cpu。
4. PyTorch v2.7.1官方batchnorm.py。https://raw.githubusercontent.com/pytorch/pytorch/v2.7.1/torch/nn/modules/batchnorm.py 。核对affine参数、运行缓冲区、train/eval统计选择、reset_running_stats与momentum=None累积平均。官方GitHub页面首次抓取失败，改读同一版本官方原始文件成功。
5. PyTorch 2.7 Reproducibility。https://docs.pytorch.org/docs/2.7/notes/randomness.html 。核对随机种子和确定性算法的作用与跨版本/平台复现边界。

文献提供背景和接口事实，不为本讲实验结果背书。图7至图11的数值来源是outputs/results.json，而非论文。验证CE、BN峰值与测试准确率均可由包内原始运行重新计算。主模型使用后激活、固定宽度、无下采样；独立投影参照为无BN、无末端门结构，不能将两者混写。
