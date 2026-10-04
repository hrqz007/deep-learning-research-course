# 008 向量内积与几何

先修：[007](../007/README.md)、[004](../004/README.md)。从分量与二维图建立内积、范数、距离、投影、余弦和高维坐标理解，不要求提前学微积分。

阅读 [讲义](lecture.pdf)、[独立实验](lab.pdf)，执行 [Notebook](experiment.ipynb)，对照 [练习详解](answers.pdf)。运行 `python experiment.py` 和 `python test_experiment.py`，需Python3.12及NumPy2.3.5，CPU离线即可。

二维例子u=(2,1)、v=(1,3)：内积5、夹角45度、投影(0.5,1.5)。三维例子提供负投影系数。其他反例区分同向与相同、单轴缩放与统一缩放、二维展示与100维坐标。

输出在outputs，运行与检查范围见verification.json。原始坐标全部合成，余弦不是概率或真实语义证据。
