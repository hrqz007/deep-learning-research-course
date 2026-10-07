# 一手来源与使用边界

核验日期：2026-10-07。以下为实际打开核对的官方文档与作者论文。正文采用原创解释、算例和图，不复制来源图。版本固定为2.14文档，安装其他版本时应再次核对API。

1. PyTorch DDP design

https://docs.pytorch.org/docs/2.14/notes/ddp.html

使用范围：梯度bucket、跨rank平均与相同更新。

2. PyTorch DDP API

https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html

使用范围：DDP假设、no_sync与用户数据划分责任。

3. PyTorch data

https://docs.pytorch.org/docs/2.14/data.html

使用范围：DistributedSampler填充、丢弃与set_epoch。

结果数字来自本单元outputs/results.json，不能由参考文档替代实测。
