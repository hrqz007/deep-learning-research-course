# DL073 数据字典

全部数据由generate_data.py原创生成，无随机数据下载、真实身份或业务记录。

| 文件/字段 | 形状或单位 | 含义 |
|---|---|---|
| data.npz: x | [10,2], float64 | 每行一条二维输入，无单位 |
| y | [10], ±1 | 类别；不是0/1概率 |
| q | [2,2] | 二次目标对角曲率1、4 |
| w0 | [2] | 二次目标初始向量[4,2] |
| separable.csv | 10行+表头 | x0,x1,y的人可读版本 |

outputs/trajectories.npz另包含quad_points[51,2]、quad_loss[51]、quad_bound[50]（对应t=1..50）；bad_points[26,2]、bad_loss[26]；run0/1/2_weights[20001,2]与对应metrics[20001,5]。metrics各列依次为平均logistic损失、参数长度、相对参考方向角度（度）、最小几何间隔、错误率。零向量的方向和间隔为NaN，表示未定义。所有结果使用float64，不含pickle对象。
