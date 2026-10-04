# 原创合成数据

三份CSV是本讲的冻结输入，无私人数据、外部数据集或预训练模型。每行是独立合成记录，id唯一；group只是生成机制中的二元审计分组，不对应真实人口属性。

字段：id为记录标识；x1、x2为模型可用实数输入；group取0或1，仅评价使用；y为0/1标签；role为train、calibration、decision或test。生成时先依次产生训练240条、验证400条、测试1000条。验证前200条固定为calibration，后200条为decision。没有按模型表现筛选数据或重新拆分。

NumPy default_rng使用PCG64，seed=3902026。每个集合先独立生成G~Bernoulli(.25)，再生成两维标准正态X，令X[:,0]+=.35G。真实logit为1.1X1−.8X2+.65X1X2−1.6+.8G，正例概率sigmoid(logit)，Y再按该概率Bernoulli抽样。每个集合顺序调用binomial、normal、binomial，然后进入下一集合。x1/x2以17位有效数字写CSV。

训练240条中74条正例；验证400条中112条正例（校准53，决策59）；测试1000条中294条正例。模型不使用G，故不能声称已获得完整P(Y|X,G)。合成机制仅用于教学，不是现实风险生成模型。

SHA-256：

- train.csv：87f4b73ac0af0229f805671b29962fbd211d2aac0a7bac127a8c96eb14e4700b
- validation.csv：a77b069b71dc5c61805978023ea079946c553a36841d3dc7c0f765b3cbf7df71
- test.csv：592b6c7814a48628fcd2afd8d68f628bd672ec9cf47d4c18c3bdf1c522b326b8

脚本首先验证摘要，CSV是复现输入的权威版本。其他NumPy版本重新生成有差异时，不得默默覆盖原文件后继续声称复现原结果。应另存数据、修改并记录协议，再重新进行完整验证。
