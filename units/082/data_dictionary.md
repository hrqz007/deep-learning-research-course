# 数据字典与观测模型

data/Misra1a.dat是原始1932字节NIST真实观测文件，14条，原始列顺序volume y、pressure x。压力范围77.6–760，体积10.07–81.78。具体压力/体积计量单位未在源文件给出。A单位同体积，b单位为压力的倒数。

拟合假设：y=A(1−exp(−bx))+ε；x固定已知，增益1、零点0，ε独立同方差高斯是用于条件不确定性分析的假设，不是源数据认证事实。没有逐点误差和标定元数据。

内存缩放：x=pressure/1000，y=volume/100；这些常数不是SI换算。参数反变换A_original=100A_scaled，b_original=b_scaled/1000。

synthetic_*.csv共24行/文件：x_dimensionless是人为实验位置，truth是A=2.4,b=0.55下无量纲真值，last_noisy_replicate仅保存对应设计第200次噪声响应，不能称作真实观测。全部200次估计见outputs/fits_*.csv的a,b列。噪声标准差0.01，种子82。

outputs/real_predictions.csv共14行，pressure_source_units、volume_source_units为源数据，full_fit使用全14点，low_pressure_fit使用低10点，linear_low_pressure_fit为低10点直线；只可用后两列进行预设高4点留出比较。

outputs/real_bootstrap.csv共300行，两个参数均已转回源单位。它是模拟参数分布，不是300次真实实验。metrics内synthetic分位数是已知真值下重复估计分布；real条件区间是围绕拟合参数的自助法，二者不要混为一谈。

数据未提供质量收支或时间信息，守恒指标不适用；零压力、单调和饱和是公式结构约束。2000压力预测属于未经观测验证的外推。
