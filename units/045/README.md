# 045 视觉输入与增强设计

先修037、043。围绕图像轴与数值、几何标签同步、插值/归一化、完整前处理梯度及可重放增强，完成固定背景变化消融。

## 材料

- [讲义PDF](lecture.pdf)及[Markdown](lecture.md)：输入含义、坐标、标签、两轮5参数完整账本、12次CNN真实实验和失败边界。
- [独立实验PDF](lab.pdf)及[Markdown](lab.md)：环境、数据、接口、重放与提交步骤。
- [练习详解PDF](answers.pdf)及[Markdown](answers.md)：12题完整解答。
- [真实执行Notebook](experiment.ipynb)、[脚本](experiment.py)、[测试](test_experiment.py)、[生成器](generate_data.py)、[12幅原创图的代码](make_figures.py)。
- [数据说明](data/README.md)、[一手来源](sources.md)、[核验范围](verification.json)。

## 运行

按environment.yml建立独立环境。实际核验Linux CPU、Python3.12.14、NumPy2.3.5、torch2.7.1+cpu、Matplotlib3.10.8。若需明确CPU版可从PyTorch官方索引安装：

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python test_experiment.py
python -O test_experiment.py
python experiment.py --output rerun
python -O experiment.py --output rerun-optimized
python make_figures.py --output rerun-figures
```

从任意工作目录可用完整路径调用脚本。默认数据按脚本位置定位；显式相对输出按当前工作目录解释。实验全离线、不用外部照片、预训练模型或付费API。

Notebook从本单元或仓库根启动，首格核对21个源码/数据/结果/图文件摘要，真正重新完成12次训练、全部1932参数状态与测试读数，保存7代码格、4幅实际生成PNG。执行器为新Python进程内真实IPython InProcessKernel，未测试浏览器UI或跨进程socket。PyTorch/NumPy数值末位在其他环境可能变化，完整跨平台字节一致未保证。

## 数据与结果范围

48训练、48验证、128测试图；四臂×三初始化种子，每次160步、297参数，合1920更新和92160训练图呈现。固定增强计划保存每一步48图的新背景与旋转标志。背景干预使用生成器已知前景mask，是额外训练信息，不假定真实应用无需分割标注。

不增强：原相关100%、反相关0%；仅背景增强：反相关100%，但原相关NLL高于无增强。正确旋转在反相关/中性NLL上未超过仅背景增强；错误旋转旧标签保留全部种子差异，没有重抽样本以追求50%答案。三测试视图是同一图像的配对干预，三种子是算法重复，均不冒充更多独立现实数据。

results.json保存精确账本、几何结果、12模型的完整测试logit/每图损失、最终参数、均值及环境。training_traces.json.gz无损保存全部1932参数状态与每步增强损失，results记录原文和压缩包摘要。每个输出文件单独原子替换，不声称多文件事务。

## 输入与重建限制

核心接口支持有限实数CHW、正标准差、半像素双线性（明确无抗混叠）、整数ID最近邻、内部裁剪加水平翻转及合法半开XYXY框；错shape/数值/坐标/尺寸/超一百万元素图输出均拒绝。实验重放只支持冻结数据协议。没有实现EXIF、颜色管理、任意透视/旋转或生产数据加载器；抗混叠例子由官方torch单独提供。

绘图需要Noto Sans CJK，系统官方包名常为fonts-noto-cjk，也可用[Google Noto CJK官方字体](https://github.com/notofonts/noto-cjk)，设置DL_CJK_FONT为文件路径；未打包字体二进制。可选PDF构建工具在shared/build-tools，含MathJax/WeasyPrint等依赖说明。未声称跨系统Anaconda新安装、GPU或浏览器Jupyter验证。

作者检查、独立验收和远端发布分别记录，实际公开状态以课程发布记录为准。
