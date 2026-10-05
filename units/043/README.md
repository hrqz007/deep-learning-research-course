# 043 卷积局部性与平移结构

先修025、027、031。从滑动窗口与参数共享走到自己实现的多通道前后向、两轮精确反传账本、平移条件检验及真实卷积核学习。

## 阅读材料

- [讲义PDF](lecture.pdf)与[Markdown](lecture.md)：互相关、形状、5参数完整手算、输入梯度、感受野、边界/步幅/任务头及真实训练。
- [独立实验PDF](lab.pdf)与[Markdown](lab.md)：环境、数据、操作步骤、独立核验与提交标准。
- [练习详解PDF](answers.pdf)与[Markdown](answers.md)：12题逐步解答。
- [已执行Notebook](experiment.ipynb)、[实验脚本](experiment.py)、[测试](test_experiment.py)、[数据生成器](generate_data.py)、[原创图生成器](make_figures.py)。
- [数据说明](data/README.md)、[来源说明](sources.md)、[检查范围](verification.json)。

## 环境与运行

按environment.yml建立独立Anaconda环境。实际执行为Linux CPU、Python3.12.14、NumPy2.3.5、PyTorch2.7.1+cpu、Matplotlib3.10.8。没有验证全部平台的新Anaconda安装、GPU或浏览器Notebook界面。

如需明确安装CPU版PyTorch，在已激活的独立环境使用官方索引：

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python test_experiment.py
python -O test_experiment.py
python experiment.py --output rerun
python -O experiment.py --output rerun-optimized
python make_figures.py --output rerun-figures
```

脚本默认输入按自身目录定位，显式相对输出路径按当前工作目录解释。数据生成器默认可重建data中的同名文件，建议用--output generated-data保留原件。数值实验离线、不下载模型、不调用付费API。

Notebook从单元目录或仓库根打开；重启内核并从头运行，首格核对18个源码/数据/结果/图文件摘要，再真实重做三次手算状态、七项平移探针与80步双实现训练。提供的Notebook保存了真实执行的8个代码单元和4幅运行时生成PNG，不依赖浏览器自动展示后端。核验采用新Python进程中的真实IPython InProcessKernel；不声称检验外进程socket传输。

绘图要求Noto Sans CJK字体。可从系统官方包仓库安装fonts-noto-cjk，或从[Google Noto CJK官方仓库](https://github.com/notofonts/noto-cjk)取得字体并设置DL_CJK_FONT为字体文件路径。未将字体二进制打包。PDF本身可直接阅读；PDF重建工具及直接依赖见shared/build-tools/README.md，要求MathJax3.2.2、WeasyPrint70与CJK字体等。PDF渲染布局随字体/库版本变化，未保证跨平台字节相同。

## 输入域与证据边界

从零算子支持有限实数NCHW、groups=1、对称零填充与正整数步幅/膨胀。拒绝非有限数、布尔/复数/对象数组、错形状、无窗口配置；原始/补边/输出数组最多100万元素，每次算子最多2000万个乘加位置，以免教学循环误用到大规模训练。运算溢出明确抛错；一般舍入与下溢仍受float64限制。不支持字符串same、分组卷积、GPU、FFT或通用周期padding接口。周期探针有独立小函数，不混入主API。

固定结果：手算均值损失0.04472656→0.03772648→0.03450670，但第一样本损失逐次变差。零边界全图平移最大误差4.5，预定内部为0；周期步幅1为0，步幅2只有输入移2格/输出移1格的对齐保证。真实学习10参数、80步，训练half-MSE0.00180414，测试0.00183973，最小二乘训练参照0.00124280。保留全部81参数状态、全部20图预测与优化不足，不宣称真实视觉优势。

作者检查、独立验收与远端公开是不同阶段；实际发布状态以课程发布记录为准。
