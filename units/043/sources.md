# 一手来源与使用范围

核对日期：2026-10-05。所有教学叙事、数据、图、代码与数值实例独立编写；没有复制论文图表或原文长段落。

1. PyTorch，Conv2d官方文档： https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.conv.Conv2d.html 。核对互相关、NCHW、stride/padding/dilation、核形状与输出公式。该动态页面本次跳到2.14文档；2.7网页抓取失败，未假称已读取该页。为锁定实际实验语义，另读取本地已安装官方torch2.7.1+cpu的torch.nn.modules.conv.Conv2d源码文档，并用该版本执行全部框架参照。官方2.7.1源码可查看 https://github.com/pytorch/pytorch/blob/v2.7.1/torch/nn/modules/conv.py 。未把最新网页版本当成本机实验版本。
2. André Araujo、Wade Norris、Jack Sim，Computing Receptive Fields of Convolutional Neural Networks，Distill，2019。 https://distill.pub/2019/computing-receptive-fields/ 。核对单路径感受野、输入坐标映射、包络空洞与路径对齐。本文公式中的dilation扩展和数值例子由相同坐标定义直接推导，并实际枚举检查。
3. Richard Zhang，Making Convolutional Networks Shift-Invariant Again，ICML 2019，PMLR 97:7324–7334。 https://proceedings.mlr.press/v97/zhang19a.html 。用于下采样/抗混叠研究背景，本讲没有复现论文真实图像成绩，不能用它证明本课探针之外的效果。
4. PyTorch 2.7 Reproducibility： https://docs.pytorch.org/docs/2.7/notes/randomness.html 。读取官方版本页，核对随机种子与跨版本、跨设备一致性边界。

更广泛背景可参阅Goodfellow、Bengio、Courville，Deep Learning第9章，作者公开站 https://www.deeplearningbook.org/contents/convnets.html 。该大页面本次工具完整抓取超限，仅搜索片段可见，因此不把它记为本次完整阅读核验来源。
