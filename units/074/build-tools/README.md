# 本单元重建

运行实验只需requirements.txt。PDF构建另需Markdown、WeasyPrint、Pango与Noto CJK字体，详见[仓库构建说明](../../../shared/build-tools/README.md)。本单元build_pdf.py沿用071的mathtext本地公式后端，支持本讲实际用到的公式。

```bash
# 从单元目录执行；先生成结果，再画图。
python generate_data.py
python experiment.py --output outputs
python make_figures.py
python create_notebook.py
python execute_notebook.py experiment.ipynb
python build_pdf.py lecture.md
python build_pdf.py lab.md
python build_pdf.py answers.md
```

重建图或结果后必须重新核验正文数字并逐页渲染阅读PDF。临时tmp、notebook_outputs和Python缓存不属于发行文件。
