# 一手来源与使用边界

核验日期：2026-10-07。以下为实际打开核对的官方文档与作者论文。正文采用原创解释、算例和图，不复制来源图。版本固定为2.14文档，安装其他版本时应再次核对API。

1. PyTorch CUDA semantics

https://docs.pytorch.org/docs/2.14/notes/cuda.html

使用范围：异步计时、同步与allocated/reserved内存语义。

2. PyTorch Profiler

https://docs.pytorch.org/docs/2.14/profiler.html

使用范围：self/total、形状与内存记录选项。

3. PyTorch Benchmark utilities

https://docs.pytorch.org/docs/2.14/benchmark_utils.html

使用范围：测量边界与重复测量。

结果数字来自本单元outputs/results.json，不能由参考文档替代实测。
