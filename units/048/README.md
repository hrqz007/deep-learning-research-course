# 048 视觉研究项目

先修022、040、044、045。按lecture.pdf、lab.pdf、experiment.ipynb学习，独立完成练习后看answers.pdf；三份Markdown源同时保留。条件先修：检测/分割补046，ViT补047，预训练补041。本单元采用从零训练CNN分类，不使用这些可选路线。

原创圆盘/圆环图像带非语义角标；24个公平预算候选、三种子、两个开发域；先保存方案与权重哈希，再打开同域与反相关两个测试集。全部轨迹、落选候选、权重、完整开发与正式测试预测均保留。四参数微型CNN逐样本前向、路径梯度相加、全部参数同步更新及下一前向，配独立标量/NumPy参照与11张原创图。

主要发现：开发规则选角标随机化。同域平均准确率与原始CNN同为95.83%；反相关域由78.52%提升到96.22%，但同域交叉熵未改善。亮度增强更差，已知位置遮蔽在固定预算下未胜出。结论仅覆盖一个合成生成器和三个初始化种子。

## 绘图和完整Notebook的字体前置条件

仅运行数值训练/测试不需要CJK字体，但make_figures.py和完整Notebook必须先有Noto Sans CJK SC。Notebook首格会导入绘图模块，因此只安装environment.yml并不足以保证Run All成功。当前已验证Linux路径为/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc；脚本优先让Matplotlib查找SC字体，未找到则从该TTC提取SC到可写临时目录，不下载字体。

Debian/Ubuntu可从系统官方仓库安装并在单元目录检查：

```bash
sudo apt-get update
sudo apt-get install fonts-noto-cjk
python -c "import make_figures; print('CJK font ready')"
```

其他系统需先安装Noto Sans CJK SC并确保Matplotlib可发现，再执行同一检查；本包没有验证其他操作系统的全新安装。若检查失败，应先解决字体/临时目录可写性，不继续完整Notebook或绘图；仍可阅读已生成PDF和运行不导入绘图模块的数值脚本。WeasyPrint、MathJax等是额外的PDF重排依赖。

在本单元目录运行：

```bash
python test_experiment.py
python -O test_experiment.py
python experiment.py --output reproduced-run
python make_figures.py --results reproduced-run/results.json --output reproduced-figures
python build_tools/execute_notebook_inprocess.py experiment.ipynb
```

不覆盖outputs；新实验必须给新目录。Notebook完整重跑并核对科学结果，图片来自本轮真实PNG。制作环境在新Python进程内创建真实InProcessKernel；未验证外进程socket或浏览器Jupyter界面。requirements.txt与environment.yml描述独立环境；notebook>=7,<8为本地浏览器可选依赖，制作环境没有安装或测试其UI。outputs/environment.json记录已执行环境的真实版本。

数据与来源：DATA_LICENSE.md、data/manifest.json、protocol.json、sources.md。数值实验无网络、API、GPU或外部预训练要求。重排PDF见build_tools/README.md，第三方软件与字体不打包。许可仅覆盖本单元原创成果，不变更其他课程或依赖许可。

本单元保持原完整102讲大纲。verification.json记录作者执行与目视范围，并不等于独立审查通过。freeze-sha256.json给出交付冻结文件索引。
