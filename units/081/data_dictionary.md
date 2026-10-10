# 数据字典

所有数值均无量纲，来源为本课已知Poisson方程。data/evaluation.csv每行一个评价位置，共1001行。

- x：闭区间[0,1]等距坐标；不是另一个PDE实例。
- truth：sin(πx)，解析真解，只用于评价。
- pinn：65配点、种子81模型预测。
- pinn_seed82：65配点、种子82模型预测。
- pinn_sparse：5配点、种子81消融预测。
- fd63：63内部节点的三对角差分解分段线性插值。

outputs/*_history.csv的step是参数更新编号，loss为更新前总目标，包含20倍边界均方。npz保存w,b,v各20个和c一个。

metrics.json中PINN的flux_balance_abs对连续值2π；差分flux_balance_discrete_abs对离散热源和，continuum_flux_error对连续值2π。时间不包括导入、文件IO、绘图和评价。不可混用守恒口径或把微小单次计时当严谨性能基准。
