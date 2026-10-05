# 一手来源与核验范围

核对日期2026-10-05。原始教学图像、数值实例、图和文字独立制作，不复制论文图表或长段原文。

1. Torchvision官方Transforms文档：https://docs.pytorch.org/vision/stable/transforms.html 。实际读取轴布局、dtype范围与image/box/mask同步变换。动态网页当前显示0.29；本课未安装或调用Torchvision，不把文档版本当作运行版本。
2. PyTorch2.7 interpolate：https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.interpolate.html 。实际读取align_corners、nearest-exact、antialias与插值约定；实验与图实际用torch2.7.1+cpu对照。自写双线性固定antialias=False；抗混叠演示由官方库单独计算。
3. Lee、Hwang、Shin（2020），Self-supervised Label Augmentation via Input Transformations，ICML/PMLR119:5714–5724：https://proceedings.mlr.press/v119/lee20c.html 。读取原始论文摘要及其关于放松不变性约束的论点，作为标签/变换关系背景。未实现其方法或复现其性能。
4. Geirhos等（2020），Shortcut Learning in Deep Neural Networks，Nature Machine Intelligence，作者预印本：https://arxiv.org/abs/2004.07780 ，DOI https://doi.org/10.1038/s42256-020-00257-z 。实际读取作者摘要与出版信息，作为训练相关线索在改变条件后失效的研究视角。不以该综述替代本课合成实验证据。

版本与执行环境见runtime_versions.json。代码不涉及外部数据下载或预训练模型；没有声称处理真实医学文件、EXIF或一般颜色管理。
