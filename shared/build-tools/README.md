# 可选的教材重建工具

阅读PDF和运行各单元实验，不需要安装这套文档构建依赖。只有希望把可信的课程Markdown重新生成PDF，或重建并检查图像时，才使用本说明。不要把构建器当作处理陌生HTML、CSS或TeX的安全沙箱。

## 已使用的构建组合

Python3.12、Markdown3.11、WeasyPrint70.0、Matplotlib3.10.8、MathJax3.2.2以及Noto Sans/Serif CJK字体。requirements.txt记录当前直接Python依赖；package.json记录当前MathJax直接依赖与XML解析包覆盖。这些文件不是完整跨平台锁文件，也没有声称执行过每种操作系统的全新安装。

WeasyPrint还需要Pango等系统组件。先按[官方平台安装说明](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html)核对自己的系统；缺字时检查中文字体是否实际可用。不要为了安装关闭证书验证或系统安全机制。MathJax的Node运行方式见[官方3.2文档](https://docs.mathjax.org/en/v3.2/server/start.html)，Node.js可从[官方入口](https://nodejs.org/en/download)获取。

## 在仓库内安装构建依赖

下列命令在仓库根目录执行。它们把Python库装进本地.deps目录，把Node库装进.build-deps。安装会访问官方包注册服务，安装前自行确认环境、下载清单和适用许可；不需要上传课程数据。

```bash
python -m pip install --target .deps -r shared/build-tools/requirements.txt
python -c "from pathlib import Path; import shutil; p=Path('.build-deps'); p.mkdir(exist_ok=True); shutil.copyfile('shared/build-tools/package.json', p/'package.json')"
npm install --prefix .build-deps
```

已有受管理的Python环境也可以把依赖装到该环境，不使用--target；构建器会优先使用存在的本地.deps，否则使用当前环境。不要把本地依赖、缓存、临时渲染页或个人文件提交到公开资料。

## 重建讲义

图文件已经随单元提供，可以直接构建PDF。若修改了数据或图，先按单元README运行make_figures.py，再执行：

```bash
python shared/build_pdf_mathjax.py units/007/lecture.md
python shared/build_pdf_mathjax.py units/007/lab.md
python shared/build_pdf_mathjax.py units/007/answers.md
```

build_pdf_mathjax.py将公式转换成自包含SVG，再由WeasyPrint输出PDF。它读取同目录build_pdf.py中的版式定义，并调用mathjax_render.cjs；三个文件需要一起保留。简单mathtext后端不支持全部矩阵环境，本课程数学正文优先用MathJax后端。

重建后必须检查排版，不以“命令成功”替代阅读。运行下面命令可把每页渲染成PNG，并附多页缩略图：

```bash
python shared/render_contacts.py units/007/lecture.pdf tmp/review-007
```

逐页检查公式、中文、图标签、表格和分页。不同系统字体与渲染库可能影响布局，不能假定另一个环境输出的PDF字节与本仓库完全相同。

## 可选Notebook核验

```bash
python shared/execute_notebook_inprocess.py units/007/experiment.ipynb
```

它在当前新启动的Python进程中创建真实IPython进程内内核，清除旧输出后顺序执行并保存。这个方式能核对代码单元顺序和变量依赖，不检验浏览器Jupyter界面或外进程通信。正常学习仍可以在JupyterLab中重启内核并运行全部。
