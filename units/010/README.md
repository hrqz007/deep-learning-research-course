# 010 导数积分与局部变化

先修：[007](../007/README.md)。从差商和累计和建立导数、积分与基本定理，不要求先学向量或矩阵。

阅读 [讲义](lecture.pdf)、[独立实验](lab.pdf)，执行 [Notebook](experiment.ipynb)，练习完整推导见 [详解](answers.pdf)。运行 `python experiment.py` 与 `python test_experiment.py`，仅需Python3.12标准库，CPU离线即可。Notebook需JupyterLab与ipykernel；重建图另需Matplotlib。

核心例子：平方函数差商、exp步长扫描、绝对值尖点、线性与平方函数积分，以及一维二次损失的有限步长。所有函数与配置为原创教学示例，不对应真实设备操作。

输出在outputs，数据与失败状态见CSV。当前实现明确区分函数定义域、未定义导数和浮点步长无法分辨，不把差分结果当作可导性的证明。运行与视觉核验详见verification.json。
