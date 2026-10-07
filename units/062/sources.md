# 一手来源与核验范围

核验日期：2026-10-07。正文以原创例子独立推导，不复制论文图片、基准数据或源码。

1. Ho、Jain、Abbeel，Denoising Diffusion Probabilistic Models，2020，v2。
   https://arxiv.org/html/2006.11239v2
   已核验第二、三节及训练/采样算法：前向马尔可夫链、边缘闭式、解析条件后验、负ELBO分解、固定方差高斯KL、ε参数化、简化MSE、末步均值展示。原论文图像量化解码器不等于本课连续二维高斯端点，正文明确区分。
2. Sohl-Dickstein等，Deep Unsupervised Learning using Nonequilibrium Thermodynamics，2015。
   https://arxiv.org/abs/1503.03585
   核验作者、标题与摘要，用于标明逐步破坏结构及学习反向过程的早期来源；不据此声称复现论文实验。
3. PyTorch 2.14，Reproducibility。
   https://docs.pytorch.org/docs/2.14/notes/randomness.html
   核对种子、确定性设置与跨版本/设备差异的限制。模型与评估随机序列分离。

本课200步日程、15→96→96→96→2网络、八团数据、三种子训练与评价阈值均为独立教学设计。outputs/results.json为数值证据。没有计算精确似然或完整ELBO；默认末步均值采样与正方差端点密度不混同。
