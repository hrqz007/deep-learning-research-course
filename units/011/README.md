# 011 偏导梯度与链式法则

先修：009 矩阵线性映射与仿射变换、010 导数积分与局部变化。

本讲回答多个参数如何共同改变标量损失。原创主例a=xy、z=a+x、L=z²在(2,3)处给出损失64、行梯度(64,32)。同一个x的两条路径必须累加48+16；沿x=t、y=t+1的路径导数则为96。全程明确多变量可微条件、行向量shape和单位方向，不预设雅可比或二阶优化理论。

## 文件与阅读顺序

1. lecture.pdf / lecture.md：12页正文、9幅原创图、12道练习。含定义、局部余项证明、方向几何、链式推导、共享路径与失败边界。
2. lab.pdf / lab.md：4页独立实验指南，任务A至I与检查点。
3. experiment.ipynb：真实顺序执行的Notebook，按手算、核心代码、失败案例与独立核验展开。
4. experiment.py：同一核心实验的脚本版。test_experiment.py：独立Fraction与80位Decimal核验。
5. answers.pdf / answers.md：5页完整详解，覆盖正文12题和实验A至I。
6. data/：原创合成配置与字段说明。figures/及make_figures.py：9幅图及可重建源码。
7. outputs/：随附脚本执行结果。Notebook单元格已保留执行输出；重新运行时另外生成notebook_outputs/，这个重复输出目录不随资料发布。source_checks.md、source-checks.json、verification.json：来源与作者验收记录。

## 环境与运行

推荐Python 3.12与NumPy 2.3.5。environment.yml同时列出Notebook需要的JupyterLab与ipykernel；核心脚本只需NumPy。运行不访问网络、API或GPU。提供的环境文件可用于学习者创建环境，但本次没有实际新建conda环境或进行跨平台安装测试。

在011目录执行以下已实际核验的核心命令：

```bash
python experiment.py --output outputs
python test_experiment.py
```

Notebook在011目录打开，选择重新启动内核并运行全部。输出写入notebook_outputs，脚本默认写outputs。两套主结果和差分表应一致。脚本的配置读取相对于experiment.py所在目录；命令行--output的相对路径则相对于当前工作目录。

生成图时另需Matplotlib与Noto Sans CJK字体，仅用于重建图片，不是核心实验依赖。make_figures.py显式寻找和注册CJK字体，缺少时会报错。重建命令为：

```bash
python make_figures.py
```

PDF构建使用课程共享MathJax与WeasyPrint离线流水线和本地依赖；学习者运行实验无需安装这些构建依赖。图与配置都是本讲原创合成内容，没有真实设备或个人数据。

## 关键结果与数值边界

- loss=64，gradient_row=[[64,32]]，shape=(1,2)
- x路径贡献48+16=64，y路径贡献32
- 单位方向(3/5,4/5)的方向导数64；速度(3,4)的路径变化率320
- x=t、y=t+1在t=2处的导数96；中心差分的精确误差12h²
- δ=(0.01,0.02)：实际损失65.28963204，线性预测65.28
- η=0.5的新点(-30,-13)，损失129600，展示大步长失败
- 连续反例在原点两个偏导0但不可微，不能无条件套方向导数公式

测试包含9组、81组Fraction夹具、32项错误输入拒绝和所有文档Python代码块。接口只承诺有限float64实数的(1,2)行输入；(2,)、(2,1)、布尔值、复数、文本与非有限数均拒绝。原始Python容器中的混合布尔值及布尔ndarray叶子会在NumPy类型提升前拒绝；已有数值ndarray只检查当前dtype，无法恢复调用前已丢失的类型来源。有限差分不自动证明可微，也不提供通用最佳步长。

## 实际执行与作者检查

已用新Python进程运行脚本与测试；Notebook使用新进程中的真实IPython InProcessKernel从空状态顺序执行，所有代码格都有真实输出且无错误格。构建沙盒限制socket，因此没有验证浏览器Jupyter界面和跨进程内核传输。脚本与Notebook的results.json、difference_scan.csv完全一致。

三份PDF共21页，已全部渲染并逐页原尺寸目视检查。来源直接检查作者教材与NumPy官方文档；正文对来源中偏导存在和可微的简略表述作了严格区分。作者验收不能替代课程发布复核，外部发布状态由总课程管理。

数值范围说明：反例函数也检查中间的欧氏范数是否有限；即使最终数学值可表示，中间范数溢出时也明确拒绝，未承诺全动态范围的稳定算法。

固定教学报告要求配置中的point为[[2,3]]，以保持t=2路径和切向标签一致；修改该项会在写输出前报错。通用的loss、gradient等函数仍可用于其他合法点。
