# 合成配置与输出字典

全部数据为原创小型微分诊断，无个人数据、真实标签、训练或性能估计。X为(2,3)，w为(3,)，b为标量()；输出种子seed为(2,3)。默认数值在config.json，schema为整数1。四组数值必须有限、绝对值≤3，并符合精确形状；拒绝bool、非有限token、重复/额外/缺失键与非零文本转float后成为0。

主图 A=X\*w+b，H=tanh(A)，Z=H\*H+H，L=sum(Z)/6。乘号均为逐元素乘法。L仅用于检查微分机制，可能为负，不是监督任务损失。固定非均匀seed定义另一个标量化目标sum(seed\*Z)，并非给均值目标更换学习率。

## 五份输出

- summary.json：配置、主目标、均值与加权梯度、独立标量参照误差、广播手算、小型完整雅可比与内积14、节点/边计数和范围
- coordinates.csv：29行，列node/index/value/mean_gradient；X6、w3、b1、a6、h6、z6、L1
- vjp.csv：10行输入坐标，给非均匀seed的VJP与独立逐坐标标量参照
- finite_difference.csv：9个步长×10坐标，共90行；节点/索引/epsilon/VJP/中心差分/绝对误差
- boundaries.csv：6个案例，分别是ReLU零点、ReLU恒等组合零点、正常三次式、带屏障三次式、多项式一阶、手工重建导数图。comparison列的含义必须连同meaning列阅读，不能当作统一“标准答案”

固定Notebook与绘图入口检查默认配置和五份输出摘要。自定义配置请用--config和--output另存，不能将固定图文静默套入新结果。严格摘要可能拒绝跨平台末位数值差异。
