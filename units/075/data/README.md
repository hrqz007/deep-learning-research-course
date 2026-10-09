# DL075 数据字典

原创无噪声函数g(x)=sin(2.4x)+0.25cos(5x)，无外部下载。输入输出无物理单位。

| 字段 | 形状 | 含义 |
|---|---|---|
| x,y | [24] | 固定训练输入与未乘幅度的g(x) |
| grid,grid_y | [151] | [-1.5,1.5]插值网格及g值 |
| train.csv | 24行 | x,y，人可读训练数据 |

outputs/trajectories.npz每组键前缀为w{宽度}_a{幅度}_s{种子}。theta0、theta、linear_delta各[3m]，顺序为a全部、w全部、b全部；nonlinear_grid、linear_grid各[151]；initial_gram[24,24]。history[51,7]每10步记录一次：步数、真实平均平方损失/2、线性化同损失、网格预测RMSE/A、相对核漂移、相对隐藏表示漂移、相对参数移动。全部float64，无pickle对象。固定训练点不是IID抽样；插值误差不冒充现实泛化风险。
