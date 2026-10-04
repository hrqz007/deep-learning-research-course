# 004 数组张量与形状思维

先修：[003](../003/README.md)。无需先学线性代数，本讲用小数组从行列与轴讲起。

读 [讲义](lecture.pdf)、[实验指南](lab.pdf)，顺序执行 [Notebook](experiment.ipynb)，练习详解见 [答案](answers.pdf)。运行 `python experiment.py` 与 `python test_experiment.py`，需要Python3.12及NumPy2.3.5，CPU即可。建议环境见environment.yml；Notebook另需JupyterLab和ipykernel。

原始数据为三条原创合成双旋钮记录。固定规则输出4、7、10，正确MAE为2/3；故意错误的两两广播平均为22/9。后者仅用于调试示范，不是正式评价分数。

输出保存在outputs，逐项数组及shape见array_ledger.json。运行检查、版本与限制见verification.json。图像通道部分只是标记数组，不使用外部图像。make_figures.py重建教学图另需matplotlib，不影响实验最小依赖。
