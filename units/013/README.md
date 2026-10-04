# 013 秩奇异值与条件数

先修：009 矩阵线性映射与仿射变换。

本讲回答“表示压缩、病态优化和低秩适配在数学上有什么共同点”。从线性相关、基、维数、秩与特征值开始，用一个2×3矩阵完整手算SVD，再推导截断误差、Frobenius最优性及可逆方阵右端扰动的条件数界。不预设导数、梯度或最小二乘；优化部分只研究平方目标的等值线。

## 文件与阅读顺序

1. lecture.pdf / lecture.md：12页正文，9幅原创图，12道自检练习。主例采用行输入xW，并明确完整与约简SVD的形状。
2. lab.pdf / lab.md：4页独立实验指南，任务A至H，包含环境、手算、数值核验、反例、失败输入与交付清单。
3. experiment.ipynb：13个真实顺序执行的代码格；含计算、解释和完整测试输出。
4. experiment.py：同一实验的可导入核心函数与独立脚本；test_experiment.py：独立Decimal/Fraction参考与失败测试。
5. answers.pdf / answers.md：4页逐题推理与实验验收答案。
6. data/：原创配置与数据字典。figures/与make_figures.py：9幅图及可重建代码。
7. outputs/：随附脚本的四个执行结果文件。Notebook保留单元格输出，重跑另外生成notebook_outputs；重复运行目录不在公开文件清单中。
8. environment.yml、source_checks.md、source-checks.json、verification.json：学习环境、逐项来源检查与作者验收记录。

## 运行

推荐Python 3.12与NumPy 2.3.5。核心脚本只依赖NumPy和标准库，无网络请求、API、GPU或在线数据。Notebook另需JupyterLab与ipykernel。环境文件已提供，但未实际新建Anaconda/conda环境，未做跨平台安装验证。

在013目录运行：

```bash
python experiment.py --output outputs
python test_experiment.py
```

这两条命令已在新的Python进程实际通过。Notebook在本目录打开，重启内核后从第一格运行全部。默认数据读取相对于experiment.py位置，--output的相对路径相对于当前工作目录；--config可显式指定另一个JSON配置。正文对应随附默认配置，修改数据后必须重新解释结论。

本次Notebook在新进程中的真实IPython InProcessKernel执行，全部13个代码格有顺序执行记录、无错误输出。执行环境限制socket，未测试浏览器Jupyter UI或跨进程内核传输。脚本与Notebook的results.json、rank_errors.csv、perturbations.csv、environment.json逐字节一致；这只证明本次同环境复现，不承诺跨平台二进制一致。

重建图形另需Matplotlib及Noto Sans CJK字体，可运行python make_figures.py。程序显式注册CJK字体，并使用DejaVu Sans补充数学字符。PDF使用课程共享MathJax与WeasyPrint构建器；实验运行不依赖PDF构建工具。

## 核验结果

- 主例W=[[3,1,0],[1,3,0]]；奇异值数学值4、2，程序为4.0、1.9999999999999996
- 秩一近似[[2,2,0],[2,2,0]]；Frobenius误差2；完全重构误差约6.66×10⁻¹⁶
- k=0、1、2时，理论Frobenius误差√20、2、0，谱误差4、2、0
- 保留80%平方能量时，相对Frobenius误差约44.72%
- ε=δ=10⁻⁶：右端相对扰动10⁻⁶，解由(1,0)变为(1,1)，相对变化1，放大10⁶倍
- diag(10,0.1)截成秩一的相对矩阵误差不足1%，但两类输出重合，标签区分信息完全丢失

测试共8组，包括100个80位Decimal解析奇异值夹具、28个Fraction非对称行系统、41项失败输入或配置拒绝、5个文档Python代码块、重复运行一致性与原输入不变。测试不会用同一个SVD函数作为自己的唯一参考。

## 限制与验收边界

核心函数只接受非空二维有限float64实矩阵；显式拒绝布尔、复数、文本、NaN、inf、无效秩预算与阈值。原始列表中的布尔在类型提升前拒绝，已有数值数组的历史类型无法恢复。数值秩阈值公开为σ>τ。condition2只接受方阵，报告浮点SVD比值；计算为零的最小奇异值返回无穷，不能据此独立证明数学奇异性。极端动态范围下还可能发生下溢；本讲不承诺全数值范围稳定性。

三份PDF共20页已全部渲染并逐页打开检查；最后修改的等值线图所在页单独重新检查，其余页像素摘要保持不变。一般SVD存在性与谱范数最佳低秩定理明确引用；自己的推导与原始来源边界分开记录。来源实际检查官方文档、作者课程与LoRA原论文第4.1节。

作者验收通过不代表独立复核或GitHub发布已经完成。独立验收和发布由课程总流程另行管理。这里的小CPU例子只验证代数与数值机制，不宣称真实任务表现、训练速度或大模型效果。
