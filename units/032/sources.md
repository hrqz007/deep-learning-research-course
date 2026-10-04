# 一手来源与实际核查范围

核查日期2026-10-04。数学推导、合成数据、完整数值trace与比较为本课程原创组织。

- S1 [Bottou、Curtis、Nocedal 作者公开论文 Optimization Methods for Large-Scale Machine Learning](https://leon.bottou.org/publications/pdf/tr-optml-2016.pdf)。实际读取第3.4节mini-batch定义、第4.1节平滑性与一次期望下降、第4.2节mini-batching权衡及步长限制、第5.2节独立样本方差关系。没有复制全文，也没有声称复现论文中的真实任务实验。本文有限总体精确枚举和标量闭式例是自己的推导与测试。
- S2 [Shamir 2016 Without-Replacement Sampling for Stochastic Gradient Methods，官方会议论文](https://proceedings.neurips.cc/paper_files/paper/2016/file/c74d97b01eae257e44aa9d5bade97baf-Paper.pdf)。实际读取摘要、引言和1.1节关于独立均匀抽样、不放回相关性与理论设定的说明。未完整审查其所有证明，不把特定凸情形保证套到本讲网络。
- S3 [PyTorch2.7 torch.utils.data](https://docs.pytorch.org/docs/2.7/data.html)。实际读取DataLoader shuffle每轮重排、sampler、batch_size、drop_last和随机生成器说明。主实验显式保存索引流，不调用DataLoader，避免把实现细节藏在框架里；未测试多worker或GPU路径。
- S4 [PyTorch2.7 Benchmark Utils](https://docs.pytorch.org/docs/2.7/benchmark_utils.html)。实际读取预热、线程控制、重复测量与中位数的基准方法说明。本讲没有调用Timer对象，而是用perf_counter_ns测量定义好的完整训练调用，协议与包含/排除开销明确记录。
- S5 [Python time文档](https://docs.python.org/3/library/time.html#time.perf_counter_ns)。实际读取perf_counter/perf_counter_ns与单调高分辨率时钟说明；访问时页面版本3.14，实际执行版本3.12.14，所用API在实际解释器中核验。无跨平台时钟实现保证。

PyTorch2.7版本与CPU执行版本2.7.1+cpu匹配。全部时间来自本机真实重复测量，不引用论文或文档的性能数字。未采用无关检索结果、二手转载、商业教材正文或私人资料。
