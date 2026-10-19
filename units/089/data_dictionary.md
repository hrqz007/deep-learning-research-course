# DL089 数据字典

## outputs/data.npz

train/val/test前缀分别为960/240/480条。images为N×8×8×3浮点图；q为N×2问题位置one-hot；h为N×2提示颜色one-hot。y为正确颜色0=red、1=blue；colors为N×2左右真值颜色；side为问题位置0=left、1=right；hint为原始提示类别。

只有images、q、h进入预测函数。y用于训练损失和评价；colors、side、hint用于生成和审计。干预器根据真值构造冲突提示属于评价权限，不能把这些字段直接传进模型。tokenize固定池化图像为N×2×5，最后两维是左右位置编码。

## 八个测试域

clean不改输入。conflict_hint把h设为必错颜色；blank_hint清空h。target_occluded清空被问半区，保留提示。target_occluded_blank_hint同时清空提示。other_occluded清空未被问半区，这种遮罩会透露目标位置。image_blank清空所有像素但保留位置编码和文字。question_flip翻转q与side，并依据colors重算y，原提示保留。

以上干预均复制数据，不修改基础测试集。除question_flip外，正确标签保持不变。

## metrics.csv与predictions.csv

指标表120行=5方法×3种子×8域。accuracy、nll是正确率与负对数似然；prediction_change_vs_clean比较同一模型对相同sample_id的类别是否变化；parameters是可训练参数；train_seconds为优化循环时间，在多个域行重复记录，汇总训练成本时不可重复相加。validation_clean_accuracy只记录不选模。attention_target_mass是注意力落在当前被问位置的平均质量，非注意力模型为null。

逐例表57600行，含方法、种子、域、sample_id、target、prediction、p_red、p_blue。不同域的相同sample_id来自同一个基础图像，统计时应保留配对结构。

## 权重

15份NPZ包含各方法实际可训练矩阵，不含pickle。w1/b1/w2/b2为答案MLP；注意力方法另含wq/wk/wv。保存文件仅支持推理复核，没有Adam和完整随机状态。JSON保存协议、环境、全部指标、训练曲线、梯度误差与重载最大误差。
