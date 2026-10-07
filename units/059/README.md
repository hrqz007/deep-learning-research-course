# DL059 自监督与对比学习

阅读lecture.pdf，按lab.pdf顺序运行experiment.ipynb，独立完成练习后看answers.pdf。讲义从两个视图到InfoNCE、温度、坍塌和冻结线性探测，比较随机、保语义对比、破语义对比和监督预训练。

## 运行

Python3.12，安装requirements.txt后进入本目录：

```bash
python test_experiment.py
python -O test_experiment.py
python experiment.py --out my_replay
python execute_notebook.py experiment.ipynb
python make_figures.py
```

Notebook真正重新训练9个350步模型，另外保留3个随机基准；保存到notebook_replay，不覆盖随包outputs。共12份编码器和12份probe。CPU、无网络数据下载。execute_notebook.py使用顺序InProcessKernel执行，未宣称测试浏览器Jupyter界面。

## 阅读结果

三种子保语义对比均值约67.80%，随机30.03%、破语义21.35%、监督99.65%。监督预训练额外用了768标签，其他预训练无标签；全部probe只用同一128标签。一个固定合成划分，不外推现实任务。详见sources.md、verification.json及outputs/protocol.json。
