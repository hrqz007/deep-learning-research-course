# DL071 分布外鲁棒性与不确定性

本单元从概率与条件平均推导不确定性、温度校准、覆盖率、选择性风险、拒答代价和子群审查。真实CPU实验训练三个网络，比较五种预定合成域，明确展示置信但错误与共享捷径。

seed17在反转域接受314/600个样本，314个全部预测错误；同一个原域校准阈值没有保护分布外性能。三个成员概率平均也在此域全错。这些是有限合成实验的证据，不是现实安全保证。

## 学习材料

- lecture.md / lecture.pdf：8页完整中文讲义，逐步数学、6张原创彩色图、实测结论与边界
- lab.md / lab.pdf：逐步运行、手算、故障植入、checkpoint复核与审查作业
- answers.md / answers.pdf：手算过程、实测计数和研究解释
- experiment.ipynb：已在新进程按序执行，真正重训并内嵌结果图
- experiment.py / test_experiment.py：完整注释代码和普通/-O均有效的16项测试

## 独立运行

在本单元目录使用独立Python环境。推荐Python3.12；已验证具体版本见verification.json。可使用Anaconda的environment.yml。首次安装依赖可能联网，实验本身完全离线，无需GPU。

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_experiment.py
python -O -m unittest -v test_experiment.py
python experiment.py --output my_run
```

outputs为教材固定证据，自己的运行建议另存my_run。默认输出名也是outputs，会覆盖同名文件。三个模型各220次全批更新，单线程CPU；通常数秒即可完成，耗时受机器与缓存影响。

## 数据和证据文件

outputs/data.npz保存800训练、400校准、600测试样本与五个配对测试域；predictions.npz保存逐例概率。results.json包含完整损失轨迹、校准候选、温度、阈值、计数、Wilson描述区间、子群结果、置信错误例和集成分歧。三个.pt是实际训练的推理权重，已全部重载核对。只加载可信文件，使用torch.load(..., weights_only=True)。没有完整优化器和随机状态，不支持精确断点续训声明。

Notebook重跑另存notebook_outputs，不覆盖固定outputs。它使用实际IPython进程内内核，未测试浏览器Jupyter界面或外进程socket通信。

```bash
python create_notebook.py
python execute_notebook.py experiment.ipynb
python make_figures.py
```

make_figures.py从固定outputs重建六图。中文图需要Noto CJK字体，Linux默认自动注册，也可通过COURSE_CJK_FONT指定本机字体文件。

## 重建PDF

```bash
python build_pdf.py lecture.md
python build_pdf.py lab.md
python build_pdf.py answers.md
```

可用build-tools/rebuild.sh串联完整流程。PDF构建需要requirements中的Markdown、WeasyPrint、Matplotlib，以及系统Pango和Noto CJK字体；阅读已有PDF不需要这些工具。公式本地生成，不调用在线渲染。

来源与适用范围见sources.md和source-checks.json；原创数据、代码和教材授权见DATA_LICENSE.md。临时渲染页、运行缓存和重复Notebook输出不属于课程交付文件。
