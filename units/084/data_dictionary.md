# 数据字典

本讲不含真实环境观测。bandit动作a=0/1，奖励为Bernoulli，均值分别0.2/0.8。两步MDP第一动作决定左/右分支，右支即时奖励−0.1，终局矩阵[[0.1,0.4],[0,1]]，折扣gamma=0.9。theta为三个状态各自的Bernoulli logit。所有奖励无量纲。

trajectories.json的a0/a1为动作；probability为当前参数下精确轨迹概率；return从时间0折扣；score为三个参数的轨迹log概率梯度。gradient_samples.npz中raw和baseline均为50000×3，同一批动作分别减0和精确J。metrics中方差为单样本梯度样本方差，standard_error为均值标准误；trained_return来自精确梯度训练。history每行为更新编号、精确J及三个参数。off_policy的ESS由重要性权重计算，非置信水平。

data/mdp.json显式保存折扣、初始logit、奖励表及转移说明。
