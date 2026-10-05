# 可选的PDF重建工具

阅读PDF、执行数值实验不需要文档构建依赖。以下命令在单元目录（例如053）执行，只处理可信的课程源码。

```bash
python -m pip install -r build-tools/requirements.txt
mkdir -p .build-deps
cp build-tools/package.json build-tools/package-lock.json .build-deps/
npm ci --prefix .build-deps
python build_pdf.py lecture.md
python build_pdf.py lab.md
python build_pdf.py answers.md
```

Python3.12、Markdown3.11、WeasyPrint70、MathJax3.2.2；要求Node、系统Pango与Noto Sans/Serif CJK。系统组件见WeasyPrint官方平台说明：https://doc.courtbouillon.org/weasyprint/stable/first_steps.html 。Node官方入口：https://nodejs.org/en/download 。只实测Linux，跨平台安装和字节一致性未验证。

build_pdf.py负责Markdown和公式处理，build-tools/base_pdf.py提供CSS，build-tools/mathjax_render.cjs调用本单元.build-deps中的MathJax。不要删除其中一个后期待独立构建。图已随包提供，变更实验时先按README重画。

重建后可运行pdftoppm -png lecture.pdf review-page，将每页实际打开检查。文本提取不能发现所有裁切、错字形或图片重叠。不要将依赖、缓存、临时渲染页或个人文件上传公开仓库。

已执行Notebook可直接阅读。python execute_notebook.py experiment.ipynb使用新进程中的in-process内核重跑；它验证顺序执行与图输出，不验证浏览器UI或外进程socket。安装实验依赖和中文字体见lab.pdf。
