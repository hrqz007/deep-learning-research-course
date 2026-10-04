# 003 函数文件与最小测试

先修：[002 Anaconda与Python第一份实验](../002/README.md)。沿用001原创合成数据，学习字典、文件、异常、函数作用域、断言、类与最小继承，不增加新的数学模型。

阅读 [讲义](lecture.pdf)、[实验指南](lab.pdf)，完成 [Notebook](experiment.ipynb)，再看 [练习详解](answers.pdf)。

在本目录执行 `python experiment.py` 与 `python test_experiment.py`。Python 3.12，实验和测试仅使用标准库；Notebook另需JupyterLab与ipykernel。CPU、离线、无需API。正式结果写入outputs，Notebook写入notebook_outputs，测试只在临时目录制造故障。

预期A训练0.2、测试0、压力测试3；B训练0、测试7。faults.py是故意错误的教学样例，不参与正式预测。不要用 `python -O`关闭测试断言。

资料是公开流程练习，不是未知盲测。实际验证方式与未验证范围见verification.json，参考核对见source-checks.json。
