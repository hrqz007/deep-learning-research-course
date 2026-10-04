# 原创合成迁移数据

没有抓取或下载第三方数据，没有用户资料、私密路径或外部权重。数据只用于教学机制与代码核验，不模拟某个真实行业数据集。

NumPy default_rng的固定种子依次是41001、41002、41003、41004。每个分割先生成N乘2的Uniform(-2,2)输入。除source外，第一坐标再乘0.875并加0.25，成为Uniform(-1.5,2)，第二坐标不变。随后生成N个独立标准差0.08的零均值正态噪声。

related标签为tanh(1.2*x1)+noise，orthogonal标签为tanh(1.2*x2)+noise。同一行的两个任务共享噪声和输入，因此两任务不是独立重采样。source文件也包含orthogonal列以保持统一格式，但源训练只使用related列。

- source.csv：512行源训练，固定600步，无独立源验证或源测试
- target_train.csv：64行目标训练池，16标签方案只取前16行
- target_validation.csv：128行选择最终候选，不再并回训练
- target_test.csv：512行，全部验证选择冻结后才解析标签并评价

CSV列是id、x1、x2、related、orthogonal；数值用17位有效数字写出，行末使用LF。ID含分割前缀。随机分割之间独立生成，不按标签排序或挑容易的点。噪声序列与输入由同一分割生成器顺序产生。

在单元目录运行“python data/generate.py --output recreated-data”可以生成新目录并与内置fixture摘要比较。原版fixture哈希在experiment.py中固定。生成器不复制原CSV。生成说明不是授权把修改后的数据混入原版结果；新研究应使用新版本与新测试。
