# 数据字典

fixture(seed=83)从固定随机数生成720行12维输入，前240行训练，接着120行新域测试，后360行旧域保留。X第一列为领域指示+1/-1，其他列是无量纲高斯上下文特征。y为0至7的下一词类别，按相应条件softmax抽样，不是真实语言token。W0为8×12原始权重，领域改动由8×2和2×12矩阵乘积生成。训练不读取测试标签。

outputs/metrics.json：adaptation_test、retention_test含nll（自然对数nat/样本）及accuracy；trainable_parameters只计更新参数；trainable_float64_bytes只计其数组存储；observed_cpu_seconds为单次实际CPU墙钟时间，不可跨机直接比较；delta_rank为数值矩阵秩；merge_max_abs_error为两路与合并logits最大绝对差。history每行step与训练NLL。weights.npz保存各方法权重。数据由源码重建，无外部下载。

data/fixture.npz保存生成后的全部划分：train_X/train_y、test_X/test_y、retention_X/retention_y、base_W，便于独立检查。
