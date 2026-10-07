# DL065 设备显存与性能测量

完整中文讲义、实验与答案，从目标和小算例推到可运行实现，保留正确分支与失败对照。

CPU单线程实际训练准确率0.92578125。向量化与逐行实现经过梯度和更新容差对照；前者本次稳态约0.527ms/step，后者约5.021ms/step，计时不是跨机器保证。逻辑参数/梯度/Adam张量载荷合计335928字节，不是峰值内存。GPU峰值字段为null。

## 阅读与运行

- lecture.pdf / lecture.md：完整教材、推导、6张原创图与有范围的结论
- lab.pdf / lab.md：逐步运行、故障植入、checkpoint复核与研究练习
- answers.pdf / answers.md：逐步推导与数值答案
- experiment.ipynb：已在新进程按序执行并内嵌PNG图

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_experiment.py
python -O -m unittest -v test_experiment.py
python experiment.py --output outputs
python make_figures.py
python create_notebook.py
python execute_notebook.py experiment.ipynb
```

在本单元目录运行。推荐独立Python3.11/3.12环境，CPU无需额外设备；首次安装依赖可能联网，实际实验不下载外部数据。完整版本与证据见verification.json。默认输出会被覆盖，扩展运行请用--output另存。Notebook重跑保存到notebook_outputs，不覆盖教材固定outputs。

## 文件与复核

experiment.py包含带注释的完整实验；test_experiment.py包含普通/-O均生效的检查。outputs/data.npz保存实际数据，outputs/results.json保留配置、完整轨迹与限制，outputs/*.pt为真正训练的终点权重。只加载可信来源，使用torch.load(..., weights_only=True)。最终权重不含完整续训状态，不能宣称严格中途断点恢复。

make_figures.py从固定结果生成6图，图中明确区分概念示意、数学估算与实测。中文绘图需要Noto CJK字体；Linux默认位置自动注册，也可用COURSE_CJK_FONT指定本机字体文件。create_notebook.py与execute_notebook.py提供重建与新IPython进程内内核顺序执行，不声称测试浏览器Jupyter界面或外进程内核通信。

## 重建PDF

```bash
python build_pdf.py lecture.md
python build_pdf.py lab.md
python build_pdf.py answers.md
```

使用当前单元的可移植Markdown/WeasyPrint/Matplotlib公式构建器，所需Python依赖在requirements.txt；还需要Noto Sans/Serif CJK与Pango等系统库。阅读已提供PDF无需构建工具。公式与图本地生成，不使用在线服务。build-tools/rebuild.sh串联重建步骤。

来源见sources.md与source-checks.json，原创数据与内容授权见DATA_LICENSE.md。缓存、依赖目录、临时渲染页与重复Notebook输出不是课程交付文件。
