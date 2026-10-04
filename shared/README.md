# 运行深度学习课程实验

001–003、006、007、010、016–022与026的核心计算只使用Python标准库。004、005、008、009、011–015、023–025与027–028使用NumPy；部分Notebook或重建图需要Matplotlib与可用中文字体，详见对应单元说明。已有Python和Jupyter环境的学习者不必为了这些实验安装深度学习框架。后续单元逐一声明额外依赖及CPU运行方式，深度学习框架在需要时再加入。

建议独立环境：在仓库根目录运行 `conda env create -f shared/environment.yml`，再运行 `conda activate dl-research-course` 和 `jupyter lab`。安装依赖通常需要网络，下载完成后的当前四十二个单元实验可离线运行。软件许可和安装权限由使用者确认。

环境文件是建议约束，不是对所有平台安装成功的承诺。各讲 `verification.json` 记录实际运行版本及限制。建议先运行脚本，再重启Notebook内核并运行全部，比较结果。

课程制作采用全新Python进程中的真实IPython InProcessKernel顺序执行；因制作环境不支持套接字，不声称验证过浏览器界面和外进程内核传输。

PDF源文件为Markdown，图与公式使用本地工具生成。重建PDF所需构建工具与学习者运行实验的依赖不同；单纯阅读PDF或运行实验不需要文档构建依赖。[可选构建工具](build-tools/README.md)包含MathJax后端、依赖说明、全页渲染和Notebook核验入口；已提供的PDF无需重建。

029–042包含实际PyTorch2.7.1 CPU实验，以各单元environment.yml和README为准；无需GPU或在线数据。
