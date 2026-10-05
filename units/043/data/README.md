# 043 原创合成数据

experiment_data.json包含三部分，均无外部图片、真实个人或私有信息。

- train_x/train_y：12幅1×6×6输入与1×4×4目标，NumPy default_rng种子4301。
- test_x/test_y：20幅同形状图，独立种子4302；固定协议不用于调参。
- symmetry_input/symmetry_kernel：种子4303的8×8整数探针及固定3×3核，专门检验平移条件。

生成输入像素独立均匀分布于[-1,1]；目标由提供的真核与真偏置0.15做valid互相关，再加独立标准差0.05高斯噪声。生成器用sliding_window_view/einsum，不调用实验中的显式卷积循环。true_kernel与true_bias仅供教学核验，不直接初始化学习模型。

NCHW的轴顺序必须保留。各图独立生成，图内相邻目标窗口共享输入，不能把所有窗口都当作独立采样单位。无缺失值或可识别信息。公开数据与答案适合流程教学，不是保密盲测。

数据字节摘要见data_manifest.json。重新生成到新目录：python generate_data.py --output generated-data。默认生成器会覆写指定目录内同名文件；不要在开始自己修改实验后无意覆盖原始输入。参考生成环境NumPy2.3.5，跨版本逐字节相同不作保证。
