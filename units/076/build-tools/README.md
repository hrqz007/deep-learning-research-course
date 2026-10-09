# 可选教材构建

阅读PDF不需要这些依赖。源码沿用课程已有build_pdf.py，使用Markdown、WeasyPrint和Matplotlib数学排版。没有在线公式服务或模型API。独立Python环境中运行：

```bash
# 安装本地文档转换库；首次下载依赖需要网络。
python -m pip install -r build-tools/requirements.txt
# 将可信课程Markdown生成同名PDF。
python build_pdf.py lecture.md
python build_pdf.py lab.md
python build_pdf.py answers.md
```

还需要系统Pango及中文字体；WeasyPrint平台依赖见[官方说明](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html)。本次使用Noto CJK字体，图生成器显式注册文件，可用COURSE_CJK_FONT指定另一中文字体路径。数学公式由本地mathtext生成SVG，正文不使用不受支持的矩阵环境。

构建后用pdftoppm逐页渲染并视觉检查。仅生成成功不能证明无缺字、溢出或重叠。tmp为构建缓存，notebook_outputs为复跑结果，不是发行文件；不要提交依赖目录或个人日志。
