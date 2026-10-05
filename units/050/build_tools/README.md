# PDF 与图表的本地重建

原始 Markdown、原始科学输出和所有图均包含在本单元中。PDF 构建无运行时网络依赖，但第一次安装环境需要从官方或知名软件仓库取得依赖。无需安装本包外部的任何私有工具。

## 运行环境

数值实验使用 ../requirements.txt 与 Python 3.12。PDF 额外使用 requirements-build.txt；MathJax 3.2.2 提供完整 TeX 转 SVG，WeasyPrint 70.0 将 Markdown 与 SVG 排成 PDF。Node.js 18 或更新版本可运行所附构建脚本。字体使用官方 Noto Sans CJK SC / Noto Serif CJK SC；Ubuntu/Debian 的 fonts-noto-cjk 提供所需字体族。

Ubuntu/Debian 可按发行版官方包源安装 fonts-noto-cjk、fontconfig、libpango-1.0-0、libpangoft2-1.0-0、poppler-utils。WeasyPrint 系统依赖随发行版不同，可参考官方安装说明 https://doc.courtbouillon.org/weasyprint/stable/first_steps.html 。macOS 可从 Noto 官方项目安装 SC 字体；本包当前自动提取逻辑仅在标准 Linux TTC 路径验证，其他平台需确保 Matplotlib 能找到对应 SC 字体族。

没有将字体二进制打包进课程。官方字体来源 https://github.com/notofonts/noto-cjk ，许可证在该官方仓库中；不要从未知字体下载站安装。

在单元目录安装与构建：

```bash
python -m pip install -r build_tools/requirements-build.txt
npm install --prefix .build-deps mathjax-full@3.2.2
export DL_MATHJAX_ROOT="$PWD/.build-deps/node_modules/mathjax-full/js"
mkdir -p .work/tmp .work/mpl .work/cache
export TMPDIR="$PWD/.work/tmp"
export MPLCONFIGDIR="$PWD/.work/mpl"
export XDG_CACHE_HOME="$PWD/.work/cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python make_figures.py
python build_tools/build_pdf_mathjax.py lecture.md
python build_tools/build_pdf_mathjax.py lab.md
python build_tools/build_pdf_mathjax.py answers.md
```

本包的 make_figures.py 优先寻找 Noto Sans CJK SC；若 Matplotlib 字体索引不能枚举 TTC 中的 SC 字体，会从 /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc 提取 SC 字体到 TMPDIR，避免修改系统字体或共享缓存。请勿忽略缺字警告后继续交付图像。

## 视觉与内容复验

使用 pdftoppm -png -r 120 lecture.pdf lecture-page 可把每页渲染成图片。逐页确认公式、上下标、中文、图例、表格、页眉页脚没有裁切或重叠。图例放在数据轴外的空白带，不能压在曲线上。使用 pypdf 提取文字只能辅助检查，不能替代逐页目视。

图表可单独输出到临时目录：python make_figures.py --out .work/regenerated-figures。源自 outputs/results.json 的数据不能以截图替代。图 6 是明示的机制示意，不是训练测量；图 13 的反事实是明示的事后诊断。

## Notebook 执行

```bash
python build_tools/execute_notebook_inprocess.py experiment.ipynb
```

此执行器在一个新的 Python 进程中创建真实 IPython 进程内核，顺序运行全部代码格并保存输出；不使用外部 socket。它不是浏览器界面测试。Notebook 通过 BytesIO 与 IPython.display.Image 嵌入实际 PNG，不靠仅在运行时有效的外部图片路径。

## 可恢复范围

恢复整个 units/050 目录即可阅读、核验数据、训练和重建图/PDF。系统字体和软件依赖按上述清单重新安装；不要求拥有原构建机器绝对路径。freeze-sha256.json 不包含自身，其余公开文件都按相对路径散列。修改后必须重新验证与冻结，不能继续声称旧冻结覆盖新文件。
