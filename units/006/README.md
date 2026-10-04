# 006 第一次可重跑的研究练习

先修：[005](../005/README.md)。读 [讲义](lecture.pdf) 与 [实验指南](lab.pdf)，按 [Notebook](experiment.ipynb) 重跑，参考 [练习详解](answers.pdf)。

Python3.12标准库即可运行 `python experiment.py` 与 `python test_experiment.py`。CPU、离线、无需API。Notebook额外需要JupyterLab和ipykernel。配置位于config.json，数据是六台虚构设备的原创合成记录。

预期主种子101选仿射规则系数2、偏移量1；验证MAE为0，教学测试MAE为2。保留三个预定种子的全部训练与验证记录。最终四条测试来自一台虚构设备，不作统计推断。

outputs是运行集合，每次在新的run-0001等目录保存配置、版本摘要、全部种子、冻结预测、逐条误差、日志与最终结果。先看run_status.json，只有complete算完整完成；失败尝试单独保留，旧结果不会被覆盖或混入。test_experiment.py含十一组检查，特别检验成功后同集合失败时新旧产物隔离。详细实际验证见verification.json。
