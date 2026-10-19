# DL091 预测世界模型与模型式决策

对应新版研究分支 M4。用二维非线性小车环境学习一步与五步动力学模型，真实执行滚动规划，并把预测误差、控制失败、部分可观测性与外推分开。纯CPU NumPy实验，完整中文讲义、实验手册、详答及六张原创图。

## 阅读顺序

1. lecture.md / lecture.pdf：状态、误差传播、多步反传、规划与研究结论
2. lab.md / lab.pdf：运行、故障植入、外推方案和验收
3. answers.md / answers.pdf：数学推导、数字例与研究审查
4. experiment.ipynb：已真实顺序执行并内嵌六图
5. experiment.py、test_experiment.py、verify_replay.py、sources.md、data_dictionary.md

## 运行

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_experiment.py
python -O -m unittest -v test_experiment.py
python experiment.py --output my_run
python verify_replay.py --output my_run
```

建议Python3.12，锁定NumPy2.3.5、matplotlib3.10.8。无GPU、外部数据、权重下载或付费API。安装依赖后全程离线，不操作任何实际硬件。默认outputs会覆盖同名证据；图位于输出目录旁figures，自定义输出也可能覆盖原图，需要保留时先复制单元。

10项测试检查梯度、递推、动作界和手算，在普通及Python -O均通过。独立重放验证18个模型/数据域组合与全部84个控制回合，重新计算真实动力学、规划动作、成本与失败。计时不含画图，通常数秒，不能视作跨硬件性能保证。

## 实测：保留负结果

五步训练小幅降低训练目标，但未稳定改善保留集预测。12个固定起点上，PD平均成本14.80，学习模型约15.63/15.65，真实模型有限搜索约16.96；这四类均通过有限时段容差。真实模型短视规划、忽略速度与零控制均未通过。正确动力学不能代替合适规划时域与观测状态。

预测比较含三个训练轨迹重采样；控制仅使用模型种子11，已明确披露。环境确定、起点有限、无真实观测或执行噪声，零失败不等于稳定性证明、安全认证或部署许可。

## 证据文件

outputs/data.npz 保存四套轨迹与动作；weights.npz 保存6个模型及部分观测回归；predictions.npz 保存所有开放环预测、全部84回合真实状态动作及代表绘图轨迹；control.csv、prediction.csv、metrics.json保存完整指标。参数足以推理重放，不包含完整续训优化器状态。

## Notebook、图与PDF

```bash
python -m pip install -r notebook-requirements.txt
python create_notebook.py
python execute_notebook.py experiment.ipynb
python make_figures.py
python build_pdf.py lecture.md
```

lab.md和answers.md可同样构建。Notebook使用新Python进程中的真实IPython InProcessKernel，顺序重训、测试、回放并内嵌PNG；未测试浏览器UI或外进程socket传输。PDF构建另需渲染依赖与中文字体，阅读现成PDF无需安装。

verification.json和outputs下的执行/回放报告是复现记录；来源与数据使用范围见sources.md、DATA_LICENSE.md。

## PDF重建依赖补充

先运行 python -m pip install -r pdf-requirements.txt。系统还需Pango共享库及Noto Sans/Serif CJK字体；Ubuntu可从官方仓库安装libpango-1.0-0、libpangoft2-1.0-0、fonts-noto-cjk。随后运行 python build_pdf.py lecture.md（lab.md、answers.md同理）。现成PDF无需安装任何渲染依赖。
