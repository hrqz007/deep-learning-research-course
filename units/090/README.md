# DL090 音频、视频与时间结构

对应新版研究分支 M3。用成对合成双音序列研究模型是否保留先后信息：从采样、混叠、Hann 窗、STFT shape、标签对齐，走到按生成源拆分的时间对照。完整中文讲义、实验手册、详答和五张原创结果图均可独立阅读。

## 阅读顺序

1. lecture.md / lecture.pdf：概念、推导、数字例、视频与音画对齐、实测与边界
2. lab.md / lab.pdf：从零重跑、故障植入、研究验收
3. answers.md / answers.pdf：完整推导和审查参考
4. experiment.ipynb：已真实顺序执行，内嵌全部五图
5. experiment.py、test_experiment.py、verify_replay.py、data_dictionary.md、sources.md

## 运行

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_experiment.py
python -O -m unittest -v test_experiment.py
python experiment.py --output my_run
python verify_replay.py --output my_run
```

建议 Python 3.12，核心依赖 NumPy 2.3.5、matplotlib 3.10.8。无 GPU、外部数据、权重下载或付费 API。安装依赖后实验全程离线。默认 outputs 会覆盖同名结果；自定义输出的图仍写到其旁 figures，需要保留原图请复制整个单元。

九组模型=三方法×三种子，全部固定600次更新，无测试集挑选种子或超参数。10项测试在普通与 Python -O 下均通过；verify_replay 重建18组有序/置乱概率数组及训练统计量。实验耗时通常数秒，保存的计时不含画图，不能当作跨硬件性能承诺。

## 实测与限制

有序时间特征模型三次均100%；池化平均约51.74%；时间模型的测试置乱平均约52.08%。置乱训练模型在有序测试出现0%、0%、100%，完整保留并解释有限随机相关性与特征分布变化，未筛种子。分组拆分源重叠为0；随机片段拆分审计共享36个源，但未据此臆造精度偏差。

480段来自40个源；只有8个保留源且每源含配对，不是96个独立现实事件。固定频带有强先验，任务不代表语音、自然视频或因果理解。

## 原始证据

outputs/data.npz 保存波形、标签、源与配对ID、分区及频带特征；weights.npz 保存权重和训练标准化统计量；predictions.npz 为逐例概率；metrics.json、results.csv、alignment.csv 提供汇总和时间审计。所有模型保存足够推理重放的参数，不宣称能原地恢复训练优化器状态。

## Notebook、图与PDF

```bash
python -m pip install -r notebook-requirements.txt
python create_notebook.py
python execute_notebook.py experiment.ipynb
python make_figures.py
python build_pdf.py lecture.md
```

lab.md、answers.md 可同样构建PDF。Notebook从新Python进程的真实IPython InProcessKernel顺序执行，完整重训并内嵌PNG；未测试浏览器Jupyter或外进程socket传输。PDF需额外渲染依赖与中文字体，阅读随课PDF不需要安装。

执行证据见 verification.json、outputs/notebook-execution.json、outputs/replay-verification.json。数据与适用范围见 DATA_LICENSE.md。

## PDF重建依赖补充

先运行 python -m pip install -r pdf-requirements.txt。系统还需Pango共享库及Noto Sans/Serif CJK字体；Ubuntu可从官方仓库安装libpango-1.0-0、libpangoft2-1.0-0、fonts-noto-cjk。随后运行 python build_pdf.py lecture.md（lab.md、answers.md同理）。现成PDF无需安装任何渲染依赖。
