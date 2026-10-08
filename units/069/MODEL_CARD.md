# DL069 教学模型卡

## 用途与范围

演示实验资产追溯和第12轮中断后的严格本地续训。它不是现实分类产品，不应用于医疗、金融、人员评价或安全关键决策。

## 模型与数据

6维float32输入，Linear(6,24)、ReLU、Dropout(0.3)、Linear(24,2)，输出两个logits。384例原创合成数据，训练256、测试128，无现实个人数据。标签规则和生成器见experiment.py及DATA_CARD.md。

## 训练与评价

Adam学习率0.01，批次32，24轮；初始化种子69，样本顺序生成器种子70。连续模型与完整恢复模型在随课固定结果中测试交叉熵0.1543545127、准确率0.9453125，逐参数和第13至24轮损失轨迹最大差为0。

## 局限

只在记录的Python/PyTorch单线程CPU环境验证；未验证跨平台、GPU、分布偏移、攻击、公平性或服务延迟。一个固定合成拆分不能估计真实应用效果。只恢复权重的分支用于失败对照，并非推荐的续训协议。

## 资产与许可

推理可使用终点checkpoint的model字段；完整续训需要optimizer、torch_rng、order_rng、epoch及配置。使用torch.load(..., weights_only=True)且仅加载可信来源。数据和原创模型许可见DATA_LICENSE.md；核心字节摘要见outputs/asset_manifest.json，完整课程文件列表见PUBLIC_FILES.json。
