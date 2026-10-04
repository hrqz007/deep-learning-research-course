# 来源核查记录

核查日期：2026-10-04。教材为独立撰写；合成数值、图、练习、程序和反例均自行设计。这里记录公式与API依据，未复制来源图片或长段文字，也不为第三方资料重新授予许可。

## 已实际打开核查的主要资料

1. Python 3.12 random 官方文档：https://docs.python.org/3.12/library/random.html
   - 查看random.Random、randrange、Notes on Reproducibility与binomialvariate段。
   - 核对有限整数范围的选择、独立实例不共享状态、seed与跨版本可复现的范围限制。
   - 本讲没有调用binomialvariate或连续均匀变量；Bernoulli生成器自行从k/M有限编号映射构造。
2. Marco Taboga，Bernoulli distribution：https://www.statlect.com/probability-distributions/Bernoulli-distribution
   - 作者公开材料，查看How the distribution is used与Definition段；明确两种结果及0/1编码。
   - 页面部分数学公式为图片，文本抽取不完整，因此公式另以下面的NIST可读表达式交叉核对，不把缺失图片当成已经检查。
3. NIST Binomial Distribution：https://www.itl.nist.gov/div898/handbook/eda/section3/eda366i.htm
   - 查看Probability Mass Function段的固定成功概率、取值范围及组合系数公式。代入n=1后系数为1，得到p与1−p。
   - 1110与成功次数3的区别由本讲独立枚举四个序列得到；未把一般二项公式当成未讲过的先修。
4. Python 3.12 math 官方文档：https://docs.python.org/3.12/library/math.html
   - 核对log1p(x)对接近0的x、fsum、exp、isfinite、prod的接口及浮点限制。
   - 代码先判定义域与端点，后调用log；不静默裁剪p。
5. NIST Normal Distribution：https://www.itl.nist.gov/div898/handbook/eda/section3/eda3661.htm
   - 查看Probability Density Function与Cumulative Distribution Function段，核对1/(σ√(2π))、平方指数、全实数值域。
   - 只使用位置和正尺度语言，不预设期望或方差，不采用页面的中心极限定理概述作为本讲论证。
6. NIST Location and Scale Parameters：https://www.itl.nist.gov/div898/handbook/eda/section3/eda364.htm
   - 查看Location Parameter、Scale Parameter、标准化公式及限制。核对f(x;a,b)=(1/b)f((x−a)/b;0,1)与b>0。
   - 平移/伸缩的面积抵消由本讲以矩形宽度独立解释。
7. NIST DLMF 7.2：https://dlmf.nist.gov/7.2
   - 实际查看7.2.1的erf积分、7.2.2的erfc积分与7.2.4无穷远值。它们共同核对全实线Gaussian积分常数。
   - 正文明确引用该标准结果，不声称已经用010内的一维工具完整证明常数闭式；C的正性/有限性另给初等上界。

8. NIST Maximum likelihood estimation：https://www.itl.nist.gov/div898/handbook/apr/section4/apr412.htm
   - 查看开头的参数函数描述与无删失观测下的密度乘积形式，用于核对离散/连续似然的角色。
   - 没有采用页面概括性的大样本性质、最优性或求偏导步骤；本讲未要求这些额外知识。

## 没有冒充验证的部分

Penn State候选概率课程页面直接打开超时，未作为正文唯一证据。NIST related distributions页存在将密度粗略称作点概率的文字，本讲未采用那种措辞；密度在本讲始终通过区间积分定义。离散似然、有限Bayes和全部原创数值例子分别用有限乘法/加法与独立Fraction重算；Gaussian点和面积分别用90位Decimal与独立积分级数核对。

未测试真实传感器的分布假设、真实样本独立性、跨平台Anaconda新安装、浏览器Jupyter界面、socket内核传输或大规模数据性能。数值积分只是有限区间近似检查，不是归一化常数的证明。
