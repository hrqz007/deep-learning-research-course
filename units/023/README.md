# 023 线性模型与最小训练闭环

中心问题：神经网络之前，怎样把模型、目标、梯度和更新接在一起？先修[009](../009/README.md)、[011](../011/README.md)、[014](../014/README.md)、[020](../020/README.md)。

## 学习材料

- [正文PDF](lecture.pdf)与[Markdown](lecture.md)：12页，9幅原创彩色图，14道练习
- [独立实验PDF](lab.pdf)与[Markdown](lab.md)：4页，手算、数组形状、梯度与收敛核对、失败实验
- [完整详解PDF](answers.pdf)与[Markdown](answers.md)：4页，覆盖正文14题与实验A至D
- [已执行Notebook](experiment.ipynb)：14个顺序代码单元
- [核心脚本](experiment.py)、[独立测试](test_experiment.py)、[环境建议](environment.yml)
- [数据字典](data/README.md)、[固定样本与拆分](data/samples.csv)、[固定配置](data/config.json)
- [来源核查](source-checks.json)、[作者验证范围](verification.json)

全部材料独立撰写，数据与图为原创合成教学材料。先看正文手算，再自己完成实验，最后读详解。

## 核心运行

本单元继续使用并复习NumPy数组。核心计算及测试只需要Python和NumPy，不需要GPU、框架、网络或账户。建议环境：

```bash
conda env create -f environment.yml
conda activate dl-unit-023
python experiment.py
python test_experiment.py
jupyter lab
```

在本单元目录启动Notebook并重启内核顺序运行。脚本默认数据路径相对脚本本身，从其他当前目录用完整脚本路径执行也可重现。

默认脚本将写入outputs/summary.json、trajectory.csv、predictions.csv、gradient_check.csv；重复运行覆盖同名结果。使用--output my_run另存；自定义数据/配置可用--samples和--config。Notebook写notebook_outputs并与随附脚本结果逐字节比较；该重跑副本不属于发布材料。

## 学到了什么

- 从分量链式法则和全微分分别推出g = A.T @ (A @ theta - y) / n，解释每个形状
- 三点手算：半MSE从11/6变成155/216；同步更新后参数为(2/3,1/2)
- 主例固定12/8/10行拆分，只用训练数据学习参数与常数2.5
- 300轮后参数约(2,−0.75,1.5)，训练/验证/测试MSE为7/60、9/320、2/25
- 对应常数基线MSE为12.3458333333、11.41875、8.6425
- 直接使用lstsq作数值参照；正规方程只用于推理，不推荐显式求逆
- 检查广播、平均因子、过大步长、未建模交互、重复与近重复特征

主例最终梯度范数约6.58e-13，训练预测与参照最大差约1.05e-12；检查点最大缩放梯度误差约3.25e-11。确定性小网格不能证明真实任务泛化，也不提供总体置信区间。

## 输入与输出契约

数值API坚持X二维、y与theta一维，拒绝布尔/复数/字符串/对象、非有限值、过大输入及明确资源上限外的配置。CSV严格检查表头、行宽、ID与拆分；JSON拒绝重复/额外键。全部输入、计算和序列化成功后才写结果，输入失败不产生新输出目录、不破坏已有文件。

输出目录链及目标文件不接受符号链接，不得与输入冲突；逐文件替换不是跨文件崩溃事务，未声称能防并发路径交换或磁盘故障。固定图/Notebook在输出前检查原始配置、数据字节摘要与重算结果，拒绝将固定解释用于别的实验。严格数值序列化比较也可能拒绝跨平台浮点末位差异；遇到这种情况应另存并核查，不删除检查后声称固定材料已复现。

## 验证与重建

作者完成12组测试：180个Fraction精确损失/梯度/五状态轨迹、100个精确有理数最小二乘解、120个独立二次展开和差分案例；另含固定拆分信息边界、非唯一参数、28类数学输入拒绝、12种坏CSV及配置/计算/序列化/路径保护。测试中的有理数消元是精确小例子审计，不是浮点稳定求解建议。

脚本在新进程和不同工作目录运行；14格Notebook由真实InProcessKernel新进程顺序执行，四个结果文件一致。三个PDF共20页均逐页目视检查，9图均已检查；最终验证细节记录于verification.json。

可选重建图需要matplotlib及Noto Sans CJK系统字体：

```bash
python make_figures.py
```

PDF从课程根目录重建：

```bash
python shared/build_pdf_mathjax.py units/023/lecture.md
python shared/build_pdf_mathjax.py units/023/lab.md
python shared/build_pdf_mathjax.py units/023/answers.md
```

构建依赖见[共享工具说明](../../shared/build-tools/README.md)，PDF重建后仍需逐页查看。建议环境不是已验证的新Anaconda安装或跨平台锁文件。作者未实测Jupyter浏览器UI、跨进程socket内核、GPU或其他操作系统安装；构建出现Fontconfig缓存权限和网络接口发现警告，但实际文件与数值均已检查。独立课程QA及远端发布状态由课程根目录清单管理。

数值转换边界另行检查：扩展精度数组/标量、CSV与JSON数值若非零却在转float64时变成0，将明确拒绝；本来就是0的输入保留。调用者若在进入接口前已自行舍入成0，接口无法恢复丢失的信息。
