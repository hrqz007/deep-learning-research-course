# 原始合成双语科研短句

generate_data.py完整决定数据，不读网络、个人资料或外部文档。train/validation/test分别48/16/24篇，训练编号0起、验证100起、测试200起，测试最后两篇含新字符探针。精确字符串不重叠，但模板、材料类别和句型有关联，不能当成自然语言独立抽样证据。

原文保留大小写、空白和Unicode形式，不做规范化。BPE只在train学习，validation/test仅编码。corpus.json中的protocol在训练前定义固定合并预算、种子、模型维度、步数和学习率。探针仅检查编码，不用于改动训练协议。

data_manifest.json给出原始corpus.json字节SHA。重新生成可指定临时目录，核对后再替换；不要在测试集上重新拟合词表。
