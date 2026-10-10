# DL079 数据字典

原创合成位置，单位m；力目标单位N；角度rad。train_x/train_y形状(192,2)，训练角度窄扇区；val_x/val_y形状(96,2)，test_x/test_y形状(256,2)，验证和测试覆盖整圆。半径均为0.3..1.8m。数据种子79/791/790，目标无测量噪声。

angles形状(61,)，含0与2π；各模型_equivariance_errors为每个角度在256位置上的最大绝对坐标差，单位N，原始值未经显示下限截断。各模型_history为每40步训练MSE，单位N²。weights.npz保存seed0的w,b,v,c；predictions.npz为seed0测试预测。results.json记录三个种子误差、相对坐标平移检查及各向异性反例。
