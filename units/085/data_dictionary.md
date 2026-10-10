# 数据字典

preferences.npz：train_X为400×2，train_y为400维；id_X/balanced_X/conflict_X各1000×2，对应y为模拟标注选择。两列为第一相对第二候选的正确性c、冗长v，均±1；y=1选第一，0选第二。训练c与v同号概率0.95；三个测试分别0.95/0.5/0。标签依sigmoid(2c)抽样。种子85用于训练，851/852/853用于测试。候选正确性直接可见是教学简化，现实无此保证。

metrics：sampled_correctness是策略对正确候选的平均概率，非一次实际抽样准确率；greedy_correctness为最大分数选择正确比例；annotator_agreement为与带噪标签一致比例；mean_reference_kl为两候选完整分布KL，单位nat；mean_verbose_probability为偏向较长候选概率。biased_judge_dpo仅训练标签改为总选长者，评价事实规则不变。history列为step、loss、两个权重。
