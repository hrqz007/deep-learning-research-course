# 可恢复构建说明

数值实验不需要本目录。重排三份PDF另需Python 3.12、Markdown 3.11、WeasyPrint 70、Matplotlib，以及Node和MathJax 3.2.2。系统需WeasyPrint支持的排版库与Noto Sans/Serif CJK SC字体；PDF检查需Poppler。第三方软件从官方发布或知名软件包仓库安装。

从独立044目录运行：

```bash
python -m pip install Markdown==3.11 WeasyPrint==70
npm install --prefix .build-deps mathjax-full@3.2.2
python build_tools/build_pdf_mathjax.py lecture.md
python build_tools/build_pdf_mathjax.py lab.md
python build_tools/build_pdf_mathjax.py answers.md
```

完整课程仓库还可复用根目录已有.build-deps；独立包默认寻找自身.build-deps。需要指定现有MathJax时，令DL_MATHJAX_ROOT指向mathjax-full/js目录。TMPDIR可指定公式SVG缓存目录。转换不请求网络、不执行Markdown里的代码。每个公式转换为自包含SVG，正文CJK字体嵌入PDF。讲义的图从figures相对路径读取。

build_pdf.py提供统一CSS，build_pdf_mathjax.py负责Markdown/公式/图注，mathjax_render.cjs负责TeX转SVG；这里是构建副本，不要求访问共享课程目录。第三方依赖未打包，安装版本列在本说明和package.json。修改文字或图后须重新渲染并目视所有PDF页面。

无socket批执行Notebook：python build_tools/execute_notebook_inprocess.py experiment.ipynb。它启动新的Python进程内核，顺序保存代码输出；并不测试浏览器Jupyter UI。绘图脚本若Matplotlib未发现TTC中的SC字形，会从已安装Noto字体集合提取SC到临时目录；不会下载字体。
