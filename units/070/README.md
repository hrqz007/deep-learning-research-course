# DL070 论文复现消融与研究表达

完整中文教材、独立实验指导与逐步答案，配6张原创概念或实测图。讲义明确区分公式示例、真实实验与未验证范围。

## 阅读与运行

- lecture.pdf / lecture.md：完整讲义与推导
- lab.pdf / lab.md：独立运行、故障注入与研究练习
- answers.pdf / answers.md：手算与实验详解
- experiment.ipynb：已在新Python进程顺序执行，内嵌真实结果图

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_experiment.py
python -O -m unittest -v test_experiment.py
python experiment.py --output my_run
```

在本单元目录运行。Python3.12、CPU已验证；不需GPU、付费API或远程数据。Linux仅CPU安装可先用官方CPU索引安装torch==2.14.1，再安装其余依赖。默认outputs会覆盖，实验扩展请用--output另存。版本、执行与限制见verification.json。只加载可信来源checkpoint，并使用weights_only=True。

## 图和Notebook

```bash
python make_figures.py
python create_notebook.py
python execute_notebook.py experiment.ipynb
```

图脚本读取固定outputs结果。Notebook实际重跑到notebook_outputs，不覆盖固定结果。执行器采用新Python进程中的IPython in-process内核，逐单元保存输出；未验证浏览器Jupyter界面。中文绘图默认注册Linux Noto CJK字体，可通过COURSE_CJK_FONT指定本机字体文件。

## 重建PDF

```bash
python build_pdf.py lecture.md
python build_pdf.py lab.md
python build_pdf.py answers.md
```

提供本单元独立Markdown/WeasyPrint/Matplotlib公式构建器；系统需Noto Sans/Serif CJK、Pango等字体渲染支持。阅读已提供PDF无需构建工具。build-tools/rebuild.sh串联全部步骤。输出目录、图、测试与模型均可离线复查；缓存和重复Notebook输出不属于交付文件。

来源与原创授权见sources.md、source-checks.json及DATA_LICENSE.md。
