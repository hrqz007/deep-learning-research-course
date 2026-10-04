# 一手来源核查

核查日期：2026-10-04。以下官方网页均在制作期间实际打开并读取相关段落。动态 stable 文档此次页面显示 scikit-learn 1.9.1；这里记录访问内容，不以该标签推断读者机器上安装的版本。数值实验不依赖 scikit-learn。

| 编号 | 实际核查范围 | 本讲用途 |
|---|---|---|
| S1 | precision_recall_fscore_support 的定义、average、labels、zero_division 与 Notes | 核对宏微平均及空分母约定 |
| S2 | Tuning the decision threshold 的分数/行动区分与验证选择警告 | 核对阈值选择也属于开发 |
| S3 | r2_score 的负值、常数目标、force_finite 与单样本说明 | 区分公式与软件默认替代值 |
| S4 | Probability calibration 的频率解释、可靠性图和 Brier 说明 | 核对校准与综合概率评分的区别 |
| S5 | Guo 等 2017 年 PMLR 论文落地页、摘要及书目信息 | 提供神经网络校准背景，不承担本文手算结论 |
| S6 | Cross-validation 中分组与时间序列拆分段落 | 核对不同未见条件的信息边界 |

S1 https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html

S2 https://scikit-learn.org/stable/modules/classification_threshold.html

S3 https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html

S4 https://scikit-learn.org/stable/modules/calibration.html

S5 https://proceedings.mlr.press/v70/guo17a.html

S6 https://scikit-learn.org/stable/modules/cross_validation.html

具体数值、合成数据、图和推导均独立编写，并由标准库有理数算术与穷举交叉核对。未复制原文段落、原图或论文实验；没有把来源列为本合成系统性能的证据。S5 仅检查落地页与摘要，没有声称完成全文 PDF 的逐页检查或复现实验。正文使用正类分箱 ECE，不将其与多分类最高置信类别 ECE 混用。
