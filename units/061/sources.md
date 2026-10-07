# 一手来源与使用边界

核验日期：2026-10-07。浏览器读取的是作者论文及官方文档；讲义推导、例题、代码、图与数据独立编写，没有复制论文图片或外部数据集。

1. Goodfellow等，Generative Adversarial Nets，2014。
   https://arxiv.org/html/1406.2661v1
   对应第3节基本目标、非饱和替代与交替更新，第4节理想判别器及JS联系。理论适用的是理想函数空间和固定G的最优D，不是本实验收敛保证。
2. PyTorch 2.14，BCEWithLogitsLoss。
   https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html
   核对logit形式二元交叉熵和数值稳定解释。代码用softplus写出两项，数学等价，不声称直接调用该类。
3. PyTorch 2.14，Reproducibility。
   https://docs.pytorch.org/docs/2.14/notes/randomness.html
   核对随机种子与确定性设置的边界；相同种子不保证跨版本、设备逐位一致。

实验自己的证据：outputs/results.json、outputs/*_samples.npz及权重。三种基准与压力配置、快照选择和最终失败状态均由程序记录。常数点对照为人工构造，独立于训练。本文未执行控制变量因果研究，相关内容明确作为练习设计。
