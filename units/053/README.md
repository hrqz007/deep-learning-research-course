# 053 Transformer块与位置

直接先修036、052。以基础算子搭建完整pre/post-norm块，逐项手算一次前向和两次同步SGD，对齐官方参考的前向、输入与全部参数梯度。再用排列、位置、未来和长度干预区分结构保证与任务能力。

- lecture.pdf/md：含完整反向链的讲义和7幅原创图
- lab.pdf/md：独立安装、7组练习、运行及验收
- answers.pdf/md：逐步数值解答与故障定位
- experiment.ipynb：已真实执行，Run All重算结果与图
- experiment.py、test_experiment.py、make_figures.py：带注释核心与验证
- outputs/results.json、runtime.json：完整三步账本、6配置对齐与结构反例
- data/protocol.json：固定协议；DATA_LICENSE.md与sources.md：数据许可和一手来源

在本目录执行：

```bash
python test_experiment.py
python -O test_experiment.py
python experiment.py --output replay
python make_figures.py --results replay/results.json --output replay/figures
jupyter lab experiment.ipynb
```

Python3.12，主要依赖见environment.yml和requirements.txt。图与Notebook需Noto Sans CJK；默认/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc，或设置DL_CJK_FONT。详见lab.pdf。只测试Linux CPU float64单线程、无dropout。测试空支持拒绝、不同长度、key padding、pre/post和参数梯度；不支持GPU/半精度/稀疏注意力。

重建PDF可用build_pdf.py，安装方法见lab.pdf；只用于可信源码。execute_notebook.py是独立新进程中的in-process内核执行，不代表测试Jupyter浏览器UI。不要删随包outputs以替换为选择性实验；新结果写replay或notebook_replay。

保留的反例：固定位置换词元不等变、固定因果mask任意换序不等变、双向前缀依赖未来；总体loss下降但一个坐标误差增加。没有pre/post优劣、语言能力或长上下文外推结论。
