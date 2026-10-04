# 原创合成数据与结果字典

本单元只有一个教学输入x=2、目标y=0.25。参数初值w=0.5、b=−0.25，模型s=(wx+b)^2+w，损失L=(s−y)^2/2。所有量无量纲。目标是检查计算图与求导机制，没有训练/验证/测试拆分，也不估计现实预测表现。

## 配置

config.json严格使用七个键：schema为整数1；x、y、w、b为有限内置数值且绝对值不超过10；iterations为0至500的整数；learning_rate在1e-6至0.1。默认40次更新、学习率0.02。拒绝bool、重复键、额外键、非有限值和文本中非零极小数转换成浮点0。合法配置不保证收敛。探索请另存--config和--output，保留默认文件供固定图文核验。

## 输出

- summary.json：配置、11个命名节点初始值/总梯度、独立展开式梯度、13节点/15边计数、最后一步和适用范围
- graph.csv：每个可达节点一行，node、operation、value、gradient；包含负号节点及常数叶子，所以比命名节点多两行
- edges.csv：每个输入位置一行，output、input、edge_index、local_partial、upstream、contribution；重复操作数保留两条边
- gradient_check.csv：w和b分别扫描12个步长，给出自动微分、中心差分、绝对误差及除以max(1,|g|,|fd|)的缩放误差
- training.csv：step=0到40，共41行；每行是在更新前参数上的prediction、loss、gradient_w、gradient_b；最后一行不再执行额外更新

backward每次返回新字典，不跨调用累加，也不自动修改参数。x、y和常数叶子的梯度是形式敏感度；训练更新清单明确只含w和b。
