# 054 预训练目标与数据构造

直接先修019、021、053。同一原创语料构造AR与简化MLM，解释编码解码可见性，逐步算NLL和两次logit更新；用未训练Transformer验证独立文档packing、位置重置、loss mask与去重。

- lecture.pdf/md：完整理论与6幅原创图
- lab.pdf/md：独立环境、8组练习、验收和排错
- answers.pdf/md：逐步推导与全部正反结果
- experiment.ipynb：真实执行；Run All重算数值与图
- experiment.py、generate_data.py、test_experiment.py、make_figures.py：带注释源码
- data/：原始11条文本、唯一10文档、词表、SHA和split
- outputs/：完整输入/标签/mask/position、loss账本及诊断结果

在本单元目录：

```bash
python test_experiment.py
python -O test_experiment.py
python experiment.py --output replay
python make_figures.py --results replay/results.json --output replay/figures
jupyter lab experiment.ipynb
```

安装见environment.yml、requirements.txt和lab.pdf。图与Notebook需Noto Sans CJK，默认/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc，或设置DL_CJK_FONT。数值核心无需字体。只实测Linux CPU float64、无dropout、单线程。原始数据由generate_data.py重建，许可见DATA_LICENSE.md。

保留的负面发现：错误未移位标签使复制规则loss近零；全局因果仍跨文档；不重置位置不等价；MLM更低的loss不代表目标更优。没有语言能力、真实去重覆盖或吞吐收益主张。PDF重建见build-tools/README.md；execute_notebook.py只核验新in-process内核顺序执行，不代表浏览器UI已测。
