# 009 矩阵线性映射与仿射变换

先修：008 向量内积与几何、004 数组张量与形状思维。

本讲用原创合成绘图台贯穿一条问题链：两项电压如何变成三个内部响应，再变成两个坐标？全程固定批次优先行向量约定：X (N,D)、W (D,H)、b (H,)或(1,H)。不要求微积分、矩阵求逆、秩或分解知识。

## 阅读与运行顺序

1. lecture.pdf / lecture.md：14页正文，9幅原创图，12道练习；从内积推出矩阵乘法，再解释基像、转置、仿射与组合。
2. lab.pdf / lab.md：4页独立实验指南，任务A到I；含完整数据、步骤、检查点与提交清单。
3. experiment.ipynb：13个已执行代码单元，重启内核后从头运行；含解释、手算检查、教学失败与输出。
4. experiment.py：同一实验的命令行版本，核心只依赖NumPy。
5. test_experiment.py：独立Fraction三层循环检查器，固定答案、64组有理数shape夹具与错误案例。
6. answers.pdf / answers.md：5页完整详解，覆盖正文12题与实验A到I。
7. data/：固定本地合成数据及字典。figures/与make_figures.py：可重建图示。
8. source_checks.md与verification.json：一手来源核对、实际运行与验收边界。

## 环境与命令

推荐Python 3.12、NumPy 2.3.5。Notebook另需JupyterLab和ipykernel；运行核心脚本不需要Jupyter。首次安装依赖需联网，实验本身不联网、不用API或GPU。

```bash
conda env create -f environment.yml
conda activate dl-unit-009
cd /path/to/009
python experiment.py --output outputs
python test_experiment.py
jupyter lab experiment.ipynb
```

在Notebook中使用Restart Kernel and Run All。不要只信已经保存的输出，也不要靠乱序执行残留变量。脚本的输入路径相对于脚本所在目录解析；Notebook应在009目录打开。生成图需另装matplotlib与Noto Sans CJK字体，仅为重建图片所需，不是核心实验依赖。make_figures.py会显式注册已找到的CJK字体；缺字体会提示错误。

```bash
python make_figures.py
```

核心环境定义不安装PDF制作依赖。课程构建时使用共享build_pdf_mathjax.py、MathJax、WeasyPrint、Poppler与本地依赖目录；这些不是学习者运行NumPy实验的前提。

## 预期结果

- Z1 = [[5,2,1],[-2,0,-1],[5,6,-2]]，shape (3,3)，单位R。
- U = [[6,0,1.5],[-1,-2,-0.5],[6,4,-1.5]]，shape (3,3)，单位R。
- Y = [[2,4.5],[-2,-1.5],[10,9.5]]，shape (3,2)，单位mm。
- Wc = [[4,3],[0,1]]，bc = [-2,-0.5]。
- 当前固定例子逐层与合成最大绝对差0；随机有理数夹具用rtol=atol=1e-12。
- 故意错误偏置在N=H=3时仍输出(3,3)，但最大绝对误差为3R。
- 测试输出status为passed，共7组检查、64组有理数夹具；另有20项拒绝输入测试和单独的错误偏置拒绝。

命令行生成outputs/report.json、positions_mm.csv与environment.json。Notebook运行时会写入notebook_outputs/同名文件，便于比较独立运行。公开资料随附脚本输出，Notebook本身保留已执行单元结果，不另附重复的notebook_outputs文件夹；两组结果都可重新生成。CSV坐标单位mm，行顺序与data/plotter.json的sample_ids一致。

## 实际验证与限制

已在Linux CPU、Python 3.12.14、NumPy 2.3.5上用新进程运行脚本和测试，并以新进程真实IPython InProcessKernel从空状态顺序执行13个Notebook代码格。输出无错误格，脚本与Notebook的report.json完全一致。构建沙盒限制socket，故未验证浏览器Jupyter界面或跨进程内核传输；未实际新建Anaconda环境，也未进行跨平台安装测试。

3份PDF最终共23页，全部渲染并逐页原尺寸目视核查。公式、数组、负号、中文、图题与偏置轴已检查。数学证明是独立逐项推导，数值证据是有限测试，两者不互相替代。课程发布/独立复核由总课程记录管理，本单元verification.json只记录作者检查。

本实验不训练模型，数据并非来自真实设备，无现实泛化、预测精度或速度结论。受控权重变化不是一般性的特征重要性；广播视图nbytes也不是新增分配量或峰值内存测量。接口只承诺非空二维有限实数数组和共享输出通道偏置。

列表与元组会在转数组前递归检查原始布尔叶子，包括混入数字的bool与np.bool_。已有ndarray只能检查当前dtype；若使用者在调用前已经将布尔值转换成整数数组，其历史类型无法恢复，不声称能据此识别原先的布尔值。
