# DL091 数据字典

## data.npz

train/validation/test/extra_states的shape分别为[48,41,2]、[16,41,2]、[32,41,2]、[32,41,2]，末轴为位置x与速度v，均为固定模拟单位。对应_actions为[N,40]；动作t将状态t推进到t+1。dt=0.15，extra同时改变初始位置和动作幅度。

## weights.npz

one_step_11等6个键为[4,2]仿射权重，设计向量[x,v,a,1]。position_only为[3]，设计[x_t,a_t,1]；position_history为[4]，设计[x_t,x_{t-1},a_{t-1},1]。

## predictions.npz

in_domain_true/extrapolation_true：[32,21,2]真实起点及前20步。in_domain_one_step_11等预测键为[32,20,2]，从真实起点自主递推，无中间真值重置。

episode_方法_回合_states为[56,2]，actions为[55]，覆盖7方法×12起点全部84回合。control_方法_states/actions是回合10的代表轨迹，专用于图，内容与相应episode键相同，不是额外样本。

## 指标

prediction.csv：方法、种子、域、全部真实输入的一步状态MSE、从初始状态开放环展开到第5/20步的MSE。两种评估的状态分布不同。

control.csv：84行，每回合初始x/v、真实累计成本、failure布尔、最大位置绝对值及终点欧氏距离。failure要求最后10个状态的|x|与|v|均≤0.15，否则为true。

metrics.json含预测时间曲线、训练五步目标、全部控制行、部分观测、配置、环境与局限。三种子来自整轨迹训练重采样；控制仅使用种子11。replay-verification.json含所有保存预测和控制重放检查；notebook-execution.json含真实Notebook执行记录。
