# 可选PDF重建

先安装实验requirements，再安装此目录requirements。系统需要Noto CJK中文字体及WeasyPrint所需字体/图形库。

```bash
python -m pip install -r build-tools/requirements.txt
python make_figures.py
python build_pdf.py lecture.md
python build_pdf.py lab.md
python build_pdf.py answers.md
```

作者使用Linux与Noto Sans/Serif CJK字体。其他平台没有逐项安装验证；重建后必须逐页检查中文、公式、图注和页脚，不应只看PDF文件存在。公式由MathText输出矢量SVG。
