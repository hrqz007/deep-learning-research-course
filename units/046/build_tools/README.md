# 可恢复构建工具

这些脚本是课程shared工具的本单元独立副本，PDF数学缓存放在TMPDIR或单元tmp中；不会改写相邻课程。实验学习不需要重建PDF。

PDF构建环境：Python 3.12、Markdown 3.11、WeasyPrint 70.0、matplotlib 3.10.8、fonttools；Node.js与MathJax 3.2.2；系统Pango及Noto Sans/Serif CJK字体；Poppler用于渲染检查。Python包可在独立环境安装，MathJax在本目录运行npm install。依赖来自官方或知名包仓库；依赖缓存不随恢复ZIP分发。

在单元目录执行：

```bash
python build_tools/build_pdf_mathjax.py lecture.md
python build_tools/build_pdf_mathjax.py lab.md
python build_tools/build_pdf_mathjax.py answers.md
python build_tools/execute_notebook_inprocess.py experiment.ipynb
```

MathJax查找顺序：MATHJAX_ROOT环境变量指向其js目录；本build_tools/node_modules；完整课程根目录.build-deps/node_modules。构建只读取本地数学包，不在渲染时访问CDN。缓存和临时文件可设置TMPDIR、MPLCONFIGDIR、XDG_CACHE_HOME、IPYTHONDIR到专属可写目录。

Notebook执行器使用新Python进程内真实ipykernel InProcessKernel，按顺序运行并保存stream与PNG，不需要socket。它没有测试浏览器Jupyter界面或跨进程通信。
