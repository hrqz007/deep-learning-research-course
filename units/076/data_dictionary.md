# T4 合成数据字典

所有字段无业务单位，仅用于一维机制教学。生成种子7601，训练100条，网格601点。生成器为experiment.py:make_data；不下载数据。

| 文件/字段 | 形状 | 含义和使用边界 |
|---|---|---|
| data.npz/train_x,train_y | 各(100,) | 输入及带噪训练观测，唯一用于拟合的数据 |
| grid_x,test_y,truth | 各(601,) | 评价输入、新噪声观测、真实均值；后两者只评价 |
| predictions.npz/means,variances | 各(3,601) | 单模型、集成、最后层工作模型，方差包含噪声 |
| members | (5,601) | 五个已训练网络均值，保留全部成员 |
| epistemic,mc_mean | 各(601,) | 固定特征参数方差、4000抽样均值 |
| traces | (5,23,2) | 成员、记录点、[步数,正则训练目标] |
| weights.npz/member_j_k | k=0,1,2为(16,)，k=3为(1,) | 成员j的隐藏w、b、输出v、c；j从0到4 |
| posterior_mean,posterior_covariance | (17,),(17,17) | 线性高斯最后层工作模型参数 |

results.json的rmse是观测误差均方根；gaussian_nll是高斯/矩匹配高斯负对数密度；coverage_95为名义95%带对观测的覆盖比例；mean_width_95是带宽均值。in_domain为|x|≤2，out_of_domain为|x|>2，far_out为|x|≥4，后者是前者子集。
