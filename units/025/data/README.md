# 合成真值表与结果字典

所有数据由本讲手工定义，不来自真实个人、机构或实验采样。samples.csv列名严格为id,x1,x2,y，四条按a、b、c、d排列。id为唯一标识；x1、x2为无量纲输入位；y为异或标签，相同输入位为0，不同为1。完整二进制定义域仅这四点。连续查询与背景图只展示网络延伸，不具有观测标签。

config.json必须含schema、affine_grid、scales、threshold_logit四个键，拒绝重复、额外、缺失键以及非有限数字或布尔数值。默认schema为1，仿射网格为−2、−1、0、1、2，输出尺度为0、1、2、4、8，logit阈值为0。所有候选使用全部四点；不存在训练/测试拆分或统计置信区间。

固定教学入口要求默认配置语义和samples.csv原始字节（包括行顺序）一致。--samples与--config适合给副本、做失败验证，不用于将任意新任务套进旧解释。通用forward、collapse_affine、binary_metrics等API可用于独立探索。

## outputs下四个文件

- summary.json：两个参数计数、尺度1ReLU指标、同权重无激活指标、合并参数、两类候选数量与最优记录、中心处分歧、有限任务范围说明。JSON拒绝NaN/Infinity。
- forward.csv：id,x1,x2,y为输入；a1,a2为隐藏激活前值；h1,h2为隐藏激活后值；logit,probability,prediction来自尺度1ReLU网络；identity_logit为去ReLU后的相同权重结果；scale8_probability为输出尺度8的概率。
- search.csv：family为affine或relu_scale；w1,w2,bias对应仿射候选，scale对应ReLU输出尺度；不适用字段为空字符串；loss为四点平均二元交叉熵，accuracy为四点正确率。先125行仿射，后5行尺度。枚举先按w1再w2再bias递增；同损失保留首个候选。
- depth.csv：17个等距t∈[0,1]，depth1、depth2、depth3分别为三角模块复合1、2、3次的值。这里只是有限采样记录，不替代连续区间上的分段证明。

自然对数，所有量无量纲。CSV用UTF-8、逗号、LF换行；所有结果在完整计算和序列化成功后才写入。浮点结果不是精确实数；严格字节检查也可能拒绝其他平台的末位差异，遇到差异应调查，不把删除检查视为复现。
