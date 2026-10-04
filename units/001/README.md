# 001 深度学习问题与证据

先修：无。只用四则运算；Python 语法在002逐步讲解。

先读 [讲义](lecture.pdf)，按 [独立实验指南](lab.pdf) 完成纸笔计算，再打开 [Notebook](experiment.ipynb)。练习完整答案见 [详解](answers.pdf)。Markdown提供可搜索、可编辑的原始文字。

## 运行

在此目录执行 `python experiment.py`。Python 3.12，标准库即可。Notebook另外需要JupyterLab与ipykernel，可使用仓库的共享环境说明。CPU、离线、不需要付费API。输出保存在 `outputs`；原始CSV不被改写。

预期：A训练 MAE 0.2，B训练0，A测试0，B测试7，A压力测试3。测试仅四个公开合成点，不是未知盲测，也不支持真实部署结论。

详细运行版本、检查记录与限制见 `verification.json`；参考资料与核对范围见 `source-checks.json`。图为本讲原创，可用 `make_figures.py` 重建，绘图另需matplotlib；运行实验不需该依赖。
