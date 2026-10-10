# DL078 数据字典

全部原创合成，无单位。data.npz中adjacency形状(150,8,8)，对称0/1邻接无原始自环；x、y形状(150,8,1)，分别节点读数与带标准差0.02噪声的响应。train_ids=0..99，val_ids=100..124，test_ids=125..149，分区单位是整图。gnn_history/blind_history为seed0每40步记录的训练MSE；smooth_variance为测试图125在0..40次纯传播后的节点方差；intervention_effect是固定该图和其他读数、对X0加1的无噪声均值变化。

weights.npz保存seed0两种模型的w,b,v,c；predictions.npz中gnn/blind形状(25,8,1)，按test_ids顺序。results.json同时保存全部三个初始化种子测试MSE；原始图、目标与分区是实验的完整可复跑输入，不依赖远程下载。
