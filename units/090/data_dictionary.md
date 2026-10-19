# DL090 数据字典

## data.npz

- waves：[480,4096] float64，单声道相对幅值，采样率8000Hz
- labels：[480] int，0为低后高，1为高后低
- groups：[480] int，生成源0–39，同源共享频率偏移与增益
- pairs：[480] int，同pair两段仅交换相同音频块
- split：[480] int，0训练、1验证、2测试，按源确定
- bands：[480,61,2] float64，两个预定频带的log1p能量相关量

## weights.npz / predictions.npz

键名前缀为方法_种子，例如 temporal_11。weights 的后缀 w、mu、sd 分别为权重与训练均值/标准差，时间特征w为[9]，池化w为[3]。predictions 每组概率为[96]，后缀_shuffled为同一模型的测试帧置乱概率；labels和groups是对应测试身份。阈值固定0.5。

## CSV和JSON

results.csv每行一个方法/种子，含验证、有序测试、置乱测试准确率。alignment.csv每行一帧，start_sample为整数起点，center_s单位秒，second_event_overlap为窗口与第二事件交集的时长比例。metrics.json另含逐源准确率、训练曲线、环境、拆分审计和局限。

replay-verification.json记录保存模型重建概率的误差。notebook-execution.json记录实际执行单元数及内核说明。elapsed_seconds_before_plotting为本机图生成前墙钟时间，不含依赖安装和绘图。所有数据由代码合成，不含个人录音。
