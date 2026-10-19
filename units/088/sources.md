# DL088 来源与核对范围

核对日期：2026-10-19。只列实际打开核对的一手来源。下面说明各来源支持什么，不代表已经复现原论文全部方法或实验。

## 1. Learning Transferable Visual Models From Natural Language Supervision

Radford et al.，2021。[论文入口](https://proceedings.mlr.press/v139/radford21a.html)。

核对材料：PMLR publication page and original paper PDF, method section / pseudocode。

已核对要点：Image/text encoders use normalized embeddings, scaled pairwise similarity, and symmetric cross entropy.

本课使用范围：Dual-encoder objective background; original small NumPy implementation, not the pretrained model or full-scale experiment.

## 2. Debiased Contrastive Learning

Chuang et al.，2020。[论文入口](https://arxiv.org/abs/2007.00224)。

核对材料：Author arXiv abstract and metadata。

已核对要点：Randomly sampled negatives can share semantic labels; the paper proposes a debiased objective.

本课使用范围：False-negative problem background only; course does not implement the paper estimator.

## 3. Supervised Contrastive Learning

Khosla et al.，2020。[论文入口](https://arxiv.org/abs/2004.11362)。

核对材料：Author arXiv abstract and metadata。

已核对要点：Supervised batch contrastive learning uses label information to bring same-class points together.

本课使用范围：Multi-positive supervision background; course-specific cross-modal target is explicitly defined in code.

## 本课原创证据

教材的手算、shape推导、合成数据生成、NumPy实现、训练权重、图表与练习由本教程编写。全部实测数字可从outputs/results.json、metrics.csv与predictions.csv追踪；引用论文只承担背景与方法来源，不替代本课证据。没有附带论文全文或第三方数据。
