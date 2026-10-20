# DL094 连续扩散与随机微分方程

原创中文教材与可复现实验。先读lecture.md，再做lab.md，最后核对answers.md。

## 运行

```bash
python -m pip install -r requirements.txt
python experiment.py
python tests.py
```

Python 3.10+，CPU运行，需要NumPy、Matplotlib。没有外部数据下载，无GPU要求。

## 文件

- lecture.md：原理、手算、边界条件、图解与练习
- lab.md：分步实验、指标解释、故障注入与验收
- answers.md：练习推导及实验解释
- experiment.py：自包含实验入口，生成真实数据、指标、三张图
- tests.py：数学恒等式、收敛或误差、保存结果验收
- outputs/data.npz：可审计数组
- outputs/metrics.json：真实运行指标
- figures/：代码生成英文标签图
- replay_output.txt：本批真实实验输出

本讲使用解析score和精确终点高斯，不训练网络；明确分开离散偏差与抽样误差。

PDF和Notebook由本批统一构建流程生成。所有图、合成数据和实验代码均为本批原创，按仓库发布流程审查。
