# DL092 多模态研究的证据与偏差

M5研究分支；先修022、071、088。以可核验合成相等判断任务，区分单模态捷径、融合表达能力、训练干预、配对置乱和跨域结论。给定带噪视觉表示，不训练图像编码器，不声称复现大型VLM。

## 阅读顺序

1. lecture.md / lecture.pdf：完整推导、五张图、实测结果与证据边界
2. lab.md / lab.pdf：手算、独立重跑、故障注入、研究报告
3. answers.md / answers.pdf：完整解答及错误推理分析
4. experiment.ipynb：已真实顺序执行，从零重训全部15个模型

## 运行

推荐Python 3.12，CPU即可。首次安装依赖需要官方软件仓库网络；实际实验完全离线，无外部模型、数据或付费API。

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_experiment.py
python -O -m unittest -v test_experiment.py
python experiment.py --output my_run
python make_figures.py
```

13项单测检查梯度、标签、置乱、确定性和损失。五种方法×三个种子×600步，共9000步全批更新；本机训练约1秒，不承诺其他机器时间。若要单线程，在运行前设置OPENBLAS_NUM_THREADS=1。默认outputs会覆盖同名证据，自己的实验推荐--output my_run。

## 文件与证据

- outputs/data.npz：训练、验证、测试基础数据与测试潜变量，仅供生成/审计
- outputs/weights.npz：15个真实训练权重
- outputs/predictions.npz：所有方法、种子、域的逐例概率
- outputs/results.json与metrics.csv：120行分组指标、主配对差、bootstrap、损失轨迹、环境
- test_output.txt：正常及python -O两次实际测试输出
- outputs/notebook-execution.json：Notebook真实执行记录

Notebook使用新进程中的IPython内核顺序执行，7个代码单元零错误；未验证浏览器Jupyter界面或跨进程socket传输。Notebook训练另存notebook_outputs，完整保留以核验，不覆盖固定教材数据。

```bash
python create_notebook.py
python execute_notebook.py experiment.ipynb
python build_pdf.py lecture.md
python build_pdf.py lab.md
python build_pdf.py answers.md
```

PDF重建额外需要Pango、Noto CJK字体和requirements中的文档包。现成PDF阅读没有这些要求。程序依赖只从官方包仓库安装；权重为普通NumPy数组，无需反序列化不可信Python对象。

## 结论范围

所有数据为原创合成，无真人或私密内容。三种子仅覆盖固定数据上的训练随机性；八域共享测试样本；bootstrap不是现实环境保证；已知提示字段与手工交互是强先验。完整来源见sources.md，许可见DATA_LICENSE.md。

## PDF重建依赖补充

先运行 python -m pip install -r pdf-requirements.txt。系统还需Pango共享库及Noto Sans/Serif CJK字体；Ubuntu可从官方仓库安装libpango-1.0-0、libpangoft2-1.0-0、fonts-noto-cjk。随后运行 python build_pdf.py lecture.md（lab.md、answers.md同理）。现成PDF无需安装任何渲染依赖。
