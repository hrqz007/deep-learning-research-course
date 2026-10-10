# DL080 独立实验册：验证可微模拟

对应S3，原先修010、011、015、029。目标是把连续解、离散解、离散梯度、连续梯度与参数恢复分开检验。预计纸笔45–60分钟，运行与分析60–90分钟。

## 1 环境与输出

本课仅需CPU，无外部数据、账户或付费API。时间单位秒，衰减率单位每秒，状态使用统一单位。热方程空间长度为1米，扩散系数0.1 m²/s。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v test_experiment.py
python -O -m unittest -v test_experiment.py
python experiment.py --output my_run
```

两种模式各应通过15项测试。results.json列出扫描列名，切勿凭列位置猜物理意义。data.npz保存ODE轨迹、敏感度、步长扫描、参数拟合记录与PDE网格；weights.npz中保存各步数估计的k，并非大型神经网络权重。

实验核心不依赖绘图库；图示与PDF是独立重建步骤。若数值测试通过但图打不开，先区分计算问题与显示问题，不要把两者混为训练失败。

## 2 纸笔方程与Euler

题1。给定dy/dt=−ky，y₀=2，k=1 s⁻¹。直接求导验证y(t)=2e⁻ᵗ，并计算T=2秒的终值。说明初值为什么不可省略。

题2。T=2，n=4，手算Euler四步状态。求终值绝对误差。解释为什么本例Euler衰减太快。把n改为8前，先预测误差是否大致减半，再运行核验。

题3。对hk=0.5、1.5、2.5分别写倍乘因子，判断稳定、非负性和准确性。为什么“幅度逐渐减小”仍可能不适合作浓度模拟？

题4。Euler一步误差通常O(h²)，为什么固定终点的全局误差通常O(h)？指出推论还需要光滑性和稳定性，而不是简单把任意误差乘步数。

## 3 梯度沿时间传播

题5。从y_(j+1)=(1−hk)y_j推导s_(j+1)=(1−hk)s_j−hy_j。若y₀不依赖k，s₀是多少？用n=4手算终点敏感度，并与闭式导数比较。

题6。写出反向λ递推以及各步对k梯度的贡献。为什么终点λ_n=1？为什么共享参数k的梯度要对每一步累加，而不是只看最后一步？

```python
from experiment import euler, reverse_gradient
k, y0, T, n = 1.0, 2.0, 2.0, 20
_, y, s = euler(k, y0, T, n)
g_reverse = reverse_gradient(k, y0, T, n)
eps = 1e-6
g_fd = (euler(k+eps,y0,T,n)[1][-1]
        - euler(k-eps,y0,T,n)[1][-1])/(2*eps)
print(s[-1], g_reverse, g_fd)
```

再调用experiment.autodiff_euler并比较返回的梯度。它用课程提供的微型计算图引擎实际执行反向自动微分。四条路径比较的是同一个离散函数，预期相近。再算连续导数−Ty₀e^(−kT)，不要要求它与粗Euler导数完全相等。如果你只检验差分与反传，却不检验连续参考，可能完全看不到求解器偏差。

## 4 步长扫描和特殊抵消

读取scan，分别画Euler状态误差、RK4状态误差与Euler梯度误差对h的双对数图。题7：本课T=2、k=1处梯度误差看起来接近二阶，能否据此宣布Euler梯度普遍二阶？按下列方式把T改成1重做探针。

```python
import numpy as np
from experiment import euler, exact
for T in [1.0, 2.0]:
    rows = []
    for n in [16, 32, 64, 128, 256]:
        _, _, s = euler(1.0, 2.0, T, n)
        error = abs(s[-1] + T*exact(1.0, 2.0, T))
        rows.append((T/n, error))
    print(T, rows)
```

用相邻误差比估计趋势：h减半，误差约减半对应一阶，约变成四分之一对应二阶。有限范围和舍入都会影响斜率；报告应写“该范围观测到”，不要把一个拟合斜率当证明。

题8。RK4每步需要四次右端计算，Euler一次。当前按相同步数比较是否等计算预算？如果要研究速度，应增加什么记录和控制？

## 5 零训练残差为什么仍会错参数

观测固定为连续真值2e⁻²，用Euler终值拟合k。题9。解方程2(1−hk_fit)^n=2e⁻²，推导k_fit=(1−e⁻ʰ)/h。计算h=0.5时的结果，解释为什么小于1。

打开fit_scan，记录n=4、32、128的估计。检查训练损失是否变小，再把估计k放回连续解析模型，比较真实连续终点。你应发现“离散拟合很好”和“连续机制参数正确”不是同一验收项。

故障注入：把敏感度递推中的旧y_j换成新y_(j+1)，观察中央差分测试。另一个更隐蔽故障是只看训练损失不查网格收敛，这可能不会触发程序异常，却会导致错误科学解释。报告要区分代码错误和方法学错误。

## 6 从ODE进入PDE

题10。一维热方程需要初值与两端边界，为什么仅给初值还不够？本课两端固定0，初始sin(πx)，总热量是否应守恒？说明热量可以通过哪里离开。

题11。Δx=0.025米、α=0.1 m²/s、μ=0.4，求Δt及100步总时间。若空间步长减半，为保持μ不变，时间步长应如何改？

题12。把μ改到0.6运行heat_explicit，预期发生什么？为什么主动拒绝明显不稳定设置比悄悄画出爆炸曲线更适合作为主实验入口？若你确实要研究失稳，应如何另建清楚标注的探针？

```python
import numpy as np
from experiment import heat_explicit
x = np.linspace(0, 1, 41)
dx = x[1]-x[0]
alpha = 0.1
dt = 0.4*dx*dx/alpha
u = heat_explicit(np.sin(np.pi*x), alpha, dx, dt, 100)
print(dt, 100*dt, u[0], u[-1])
```

初始sin(πx)默认x按1米长度归一化。若杆长不是1米，应写sin(πx/L)，解析衰减指数也需要L²，不能只改网格端点。

## 7 复跑Notebook与作图

```bash
python create_notebook.py
python execute_notebook.py experiment.ipynb
python make_figures.py
```

Notebook执行会另写notebook_outputs，并运行额外T=1/2梯度探针。课程图读取outputs，不自动使用my_run。执行器实际验证新Python进程内顺序代码运行，不声称Jupyter浏览器UI已测试。

## 8 研究报告模板与验收

报告依次写：方程和单位；初边值；解析参考；Euler与RK4离散化；步数与右端调用次数；状态误差；离散梯度三方核验；连续梯度误差；参数恢复的网格偏差；PDE边界与稳定条件；适用限制。

最后回答：如果把衰减率换成神经网络，当前哪些验证仍必须保留？一个合格答案不会说“自动微分会解决所有数值问题”。它应该保留求解器误差、梯度对象、步长扫描、观测模型和传统基线的检查。
