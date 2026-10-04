# 第028单元 自动微分机制与边界

先修[027 批次反向传播与损失归约](../027/README.md)、[014 浮点数与稳定数值计算](../014/README.md)。中心问题：框架替我们求导时做了什么，又没有保证什么？

## 阅读与运行

- [正文11页](lecture.pdf)：11幅原创机制图；符号/差分/自动微分比较、完整VJP手算与组合推导、广播转置归约、共享节点、非光滑反例、屏障和高阶图
- [独立实验3页](lab.pdf)和[完整详解4页](answers.pdf)，15题与实验A至E，均有Markdown源
- [Notebook](experiment.ipynb)：15格实际执行，3张内嵌图
- [NumPy引擎](engine.py)、[实验脚本](experiment.py)、[13组测试](test_experiment.py)、[数据字典](data/README.md)、环境及五份结果

在本单元目录运行：

```bash
python experiment.py --output outputs
python test_experiment.py
```

核心和测试需要Python与NumPy。Notebook/绘图环境建议：

```bash
conda env create -f environment.yml
conda activate dl-unit-028
jupyter lab
```

重启内核，顺序运行Notebook，五份notebook_outputs与随附outputs逐字节比较。实际Python3.12.14、NumPy2.3.5、matplotlib3.10.8；真实InProcessKernel在新进程执行。全新Anaconda、浏览器Jupyter、外部socket内核、跨平台与GPU未测试。此单元没有执行或依赖JAX/PyTorch，只查官方资料。环境文件是安装配方而非验证过的锁文件。

## 核心机制与默认结果

Tensor用对象身份区分节点，复制输入并保存只读快照，value给出防御性副本。有序父边保存pullback，保留重复输入位置。vjp每次返回新字典，不自动更新参数或跨调用累计；每个返回梯度shape必须等于对应节点。标量输出可省略seed，任何向量输出都要求精确同shape种子。

支持逐元素加/乘/负号、tanh、ReLU、按轴sum及detach屏障；支持兼容广播与NumPy左侧数组的反向运算符分派。不实现矩阵乘法、可微索引、一般幂、GPU或自动高阶图。普通NumPy和逐坐标参照函数只用于已验证输入的检查，不承担公开入口校验。

主图X(2,3)、w(3,)、b()→A=X\*w+b→H=tanh(A)→Z=H\*H+H→L=sum(Z)/6。L是可能为负的微分检查目标，不是现实预测损失。默认L≈0.1359075261，均值gw≈(−0.04115095,0.20661271,0.16250020)，gb≈0.81838672。给非均匀seed后gb≈−0.20840060，目标标量化方式已不同。

手算广播得到u梯度[[6],[15]]、v梯度[[5,7,9]]。二维雅可比例的VJP=(-4,8)，JVP=(3.5,−3.5)，双边内积14。

ReLU(x)−ReLU(−x)恒等于x，普通导数处处1；在0处按两次ReLU局部零约定组合却返回0。detach(x²)\*x在x=2处前向8、返回4，而普通x³导数12。二阶5.5由手工写出的导数表达式重新建图得到，不能宣传为本引擎自动支持高阶求导。

## 失败与数值边界

引擎接受有限实数标量或非空数组，最多4轴、100000坐标、绝对值≤1e100，图最多10000节点；这些是各项上限，不是总内存预算。拒绝bool、复数、字符串、对象数组、非法轴/shape、非有限值、非零输入转float64后变零以及非零乘积下溢成零。加法舍入、抵消和小量吸收仍可能发生；入口前已丢失的信息无法恢复。

tanh输入绝对值≤100。JSON配置须有schema1与固定shape的X/w/b/seed，绝对值≤3；拒绝重复、缺失、额外键、bool、非有限token与文本非零转零。--config custom.json --output custom_outputs可另存探索。

固定图/Notebook先核对默认配置与五输出摘要；严格摘要可能拒绝跨平台末位差异。所有输入、计算和序列化先于输出写入；失败保留旧结果并不创建新目录。有效运行覆盖同名文件，不承诺OS多文件崩溃原子性或恶意并发路径防护。PDF复建见[共享说明](../../shared/build-tools/README.md)，仍需逐页核验。

## 已完成的作者检查

13组测试覆盖360个Fraction广播图、14种独立索引归约、40个6×10完整差分雅可比及内积对偶、121个100位Decimal激活点、49个精确一阶/手工二阶点、小型手算、2201节点迭代链、seed/身份/别名/NumPy反向运算符和数值输入拒绝。

五结果跨普通/-O新进程异工作目录与Notebook相同；32次坏配置或真实计算下溢CLI新旧输出保护、12次固定图/Notebook守卫、3段完整文档示例与1段局部表达式核验通过。三份PDF共18页和11图均逐一查看，修正了深色热图文字对比及Markdown乘号误解析。六个官方来源已实际核查，记录在source-checks.json。独立验收与远端状态由根清单管理。
