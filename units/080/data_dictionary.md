# DL080 数据字典

ODE时间t单位s，状态y为统一浓度单位，k单位s^-1；敏感度单位为状态乘秒。data.npz中t/euler_y/exact_y/sensitivity对应n=16的轨迹。scan每行一个步数设置，七列含义在results.json的scan_columns；fit_scan两列为步数n与估计k。fit_history_n每行是一次优化后的k及该更新前损失，二者不应混称同一时刻损失。

stability_trajectories三行对应hk=0.5、1.5、2.5，列为离散步数0..12，不同h意味着终止物理时间不同，图只比较离散稳定性。heat_x单位m；heat_u/heat_exact为统一温度差单位，L=1m，alpha=0.1m²/s，dx=0.025m，dt=0.0025s，100步至0.25s，固定零边界。

weights.npz为不同n拟合的k数组，不是深网权重；predictions.npz保存n=16 ODE及热方程预测。全部参考由解析函数或确定性模拟生成，无真实仪器数据。
