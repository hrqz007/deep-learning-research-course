# DL089 来源与核对范围

核对日期：2026-10-19。只列实际打开核对的一手来源。下面说明各来源支持什么，不代表已经复现原论文全部方法或实验。

## 1. Visual Instruction Tuning

Liu et al.，2023。[论文入口](https://arxiv.org/abs/2304.08485)。

核对材料：Author arXiv abstract and HTML sections 4.1 and 4.2。

已核对要点：The early LLaVA architecture uses a trainable projection to map visual features into language embedding width; training stages differ in updated modules.

本课使用范围：Projection-interface reading; no LLaVA weights or instruction data are used.

## 2. Flamingo: a Visual Language Model for Few-Shot Learning

Alayrac et al.，2022。[论文入口](https://arxiv.org/abs/2204.14198)。

核对材料：Author arXiv abstract and HTML sections 2.1/2.2。

已核对要点：Flamingo uses a visual resampler and newly trained gated cross-attention modules to condition a frozen language model on visual features.

本课使用范围：Cross-attention connection example; course implements only a tiny single-head classifier, not Flamingo.

## 3. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

Li et al.，2023。[论文入口](https://arxiv.org/abs/2301.12597)。

核对材料：Author arXiv abstract and metadata。

已核对要点：BLIP-2 bridges frozen pretrained image encoders and language models with a lightweight Querying Transformer trained in two stages.

本课使用范围：Further reading on query-based interfaces; no numerical comparison with BLIP-2.

## 4. OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework

Wang et al.，2022。[论文入口](https://arxiv.org/abs/2202.03052)。

核对材料：Author arXiv abstract and metadata。

已核对要点：OFA frames diverse cross-modal and unimodal tasks in a sequence-to-sequence framework.

本课使用范围：Unified-interface reading; the course does not train an OFA or unified Transformer model.

## 5. Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering

Goyal et al.，2017。[论文入口](https://arxiv.org/abs/1612.00837)。

核对材料：Author arXiv abstract and metadata; first submission 2016, revised 2017。

已核对要点：The study counteracts language priors with complementary image pairs yielding different answers to the same question.

本课使用范围：Motivates visual-necessity checks and language-only baselines; the course uses original synthetic data.

## 本课原创证据

教材的手算、shape推导、合成数据生成、NumPy实现、训练权重、图表与练习由本教程编写。全部实测数字可从outputs/results.json、metrics.csv与predictions.csv追踪；引用论文只承担背景与方法来源，不替代本课证据。没有附带论文全文或第三方数据。
