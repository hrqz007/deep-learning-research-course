# DL088 数据字典

## outputs/data.npz

train/val/test前缀分别表示训练432、验证108、测试216条。x为N×192浮点像素，对应8×8×3；t为N×10文本属性，0:3颜色、3:6形状、6:8位置、8:10模板。y为0至17语义类，解码为color=y//6、shape=(y%6)//2、side=y%2。captions为可读字符串。

x与t进入两个编码器；y只用于语义多正例训练和评价，普通单向/双向目标不使用语义关系。pair ID是同一分区的数组索引，没有作为输入特征传给编码器。测试标签不会参与训练。

## metrics.csv

method、seed、tau、steps、batch记录配置；parameters为3232；seconds是优化循环墙钟秒数。i2t/t2i分别为图到文/文到图，pair是唯一原配对金标准，semantic是任一同类候选。r1/r5为前k内至少命中一个正确候选的查询比例；pair_mrr为原配对倒数排名均值。validation_semantic_r1只记录验证表现，不用来选择模型。reload_max_error比较保存重载前后的测试相似度。

## predictions.csv

9模型×2方向×216查询=3888行。query_id、top_id是测试数组索引；query_class、top_class是语义类；pair_correct与semantic_correct按两种定义给0/1。这里只保存Top-1，完整排序可以从权重与data.npz重算。

## 权重与JSON

每个主模型NPZ包含wi（192×16）和wt（10×16），不含pickle。9份主权重支持推理重载，不含Adam状态，不能称为完整训练恢复快照。两次温度诊断只存指标，默认不提供其权重。

JSON含协议、环境、审计、三个种子全部记录、训练曲线、温度诊断及总时间。图像近邻余弦是像素空间诊断，不是训练后语义相似度。
