# DL088 跨模态表示与对齐

对应新版大纲M1。真实训练双编码器，检查单向/双向目标、语义假负样本和两种检索金标准。

三种子实测语义R@1约99.4%至100%，严格实例配对R@1约6%至10%。同义候选和缺失实例信息使两者不同；双向目标没有在本窄任务稳定优于单向。

## 材料

- lecture.md / lecture.pdf：完整中文教材，21节、5张原创图、推导、数值例与实测讨论
- lab.md / lab.pdf：重跑、手算、代码检查、干预与研究扩展任务
- answers.md / answers.pdf：数值答案与审查解释
- experiment.py：独立CPU NumPy实现，全部梯度手写
- experiment.ipynb：已执行Notebook；真实重训到notebook_run，不覆盖发行outputs
- test_experiment.py：15项测试，普通模式与python -O均通过
- research_plan.json、project_report.md、data_dictionary.md：协议、完整短报告和字段说明
- outputs/：真实数据、权重、JSON/CSV指标、逐例预测、测试与Notebook执行记录
- figures/：5张可独立打开的PNG
- requirements.txt、environment.yml、DATA_LICENSE.md：依赖和数据来源

## 运行

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python experiment.py
python -m unittest -v test_experiment.py
python -O -m unittest -v test_experiment.py
python execute_notebook.py
```

Windows PowerShell可用$env:OPENBLAS_NUM_THREADS="1"设置单线程后再运行python。只运行实验与测试需要NumPy和Matplotlib；Notebook生成和回放还需要requirements中的Jupyter依赖。已验证Python3.12.14、NumPy2.3.5、Matplotlib3.10.8。

默认脚本会重写outputs和figures中的同名文件。保留发行证据时请复制单元目录。Notebook在notebook_run生成另一套结果，已保存回放使用新的Python进程、in-process IPython内核顺序执行全部单元；没有声称测试浏览器Jupyter界面。

本单元可单独解压运行，无需上一讲隐藏状态。训练不下载数据或权重，不调用收费API。秒数随机器变化，固定种子不能保证跨版本与平台逐位相同。检查参数与指标，不要只比较计时。

## 边界

文本输入是预先定义的10维属性向量，图像是8×8合成图；这不是自然语言编码器或CLIP规模复现，也不评估未见概念零样本泛化。多正例方法额外使用生成器语义标签。

## PDF重建依赖补充

先运行 python -m pip install -r pdf-requirements.txt。系统还需Pango共享库及Noto Sans/Serif CJK字体；Ubuntu可从官方仓库安装libpango-1.0-0、libpangoft2-1.0-0、fonts-noto-cjk。随后运行 python build_pdf.py lecture.md（lab.md、answers.md同理）。现成PDF无需安装任何渲染依赖。
