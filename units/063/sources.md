# DL063 一手来源与核验范围

核验日期2026-10-07。文献用于概念归属和方法背景，实验数据与图均为本课原创；不复制第三方图或论文全文。

1. [Auto-Encoding Variational Bayes](https://arxiv.org/html/1312.6114v11)
   - 核验方式：Opened primary full text, sections2 and3 and appendixB
   - 使用范围：VAE probability model, ELBO, reparameterization and diagonal Gaussian KL

2. [PyTorch2.14 Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)
   - 核验方式：Opened official versioned documentation
   - 使用范围：Random seeds and deterministic settings do not guarantee identity across releases/devices

3. [Generative Adversarial Nets](https://arxiv.org/html/1406.2661v1)
   - 核验方式：Opened primary full text
   - 使用范围：Original GAN training and non-saturating generator objective

4. [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/pdf/1706.08500)
   - 核验方式：Opened primary PDF
   - 使用范围：FID attribution and fitted-Gaussian feature distance

5. [Assessing Generative Models via Precision and Recall](https://arxiv.org/abs/1806.00035)
   - 核验方式：Opened primary abstract
   - 使用范围：Separate quality and coverage as conceptual dimensions; this lesson does not implement that estimator

6. [Effectively Unbiased FID and Inception Score and where to find them](https://arxiv.org/pdf/1911.07023)
   - 核验方式：Opened primary PDF
   - 使用范围：Finite-sample bias can affect FID estimates; original synthetic sample-size experiment

7. [The Secret Sharer](https://arxiv.org/abs/1802.08232)
   - 核验方式：Opened primary abstract
   - 使用范围：Memorization risk background only; no implementation of exposure or privacy attack

## 证据分层

论文中的一般方法结论与本课实际测量分开。数值表来自outputs/results.json，完整配置来自outputs/protocol.json，图由make_figures.py读取保存数组生成。标准概率恒等式与小型手算例子在讲义内独立推导。摘要核验的文献只用于其明确陈述的研究主题，不据此声称验证全文所有定理。
