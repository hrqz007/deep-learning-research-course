# 来源核查记录

核查日期：2026-10-04。仅列直接打开并用于本单元定义或接口核对的一手论文、作者公开教材、官方文档。正文、合成数据、有限三候选例子、代码、图和练习均独立制作，不转载原图或大段文字。核查不能替代本文每步推导与程序验证。

1. Claude E. Shannon，1948，A Mathematical Theory of Communication，作者论文的Harvard公开托管PDF。
   - URL：https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf
   - 实际打开：55页PDF的可检索正文；定位第6节“Choice, Uncertainty and Entropy”，PDF第10至11页及开头的对数单位讨论。
   - 核对：有限概率的熵形式、零熵与确定性、均匀分布最大熵、对数底改变单位。
   - 限制：网页PDF文本提取会丢失部分负号，未将提取后的公式直接复制；本文符号经独立手算及其他作者来源交叉核对。另一次网页截图尝试返回不可用，未据此宣称完成来源PDF视觉审阅；本课程自己生成的全部PDF另有逐页视觉检查。

2. Ian Goodfellow、Yoshua Bengio、Aaron Courville，Deep Learning，作者网站第3章第3.13节。
   - URL：https://www.deeplearningbook.org/contents/prob.html
   - 实际打开：HTML正文，定位式3.48至3.51及图3.6周围解释。
   - 核对：自信息、熵、KL的期望方向、非对称性、交叉熵分解及零权重约定；作者的模式示例有明确受限近似族，不能无条件外推。
   - 使用边界：本讲以三个固定有限候选重建不同方向的比较，不采用原书连续Gaussian图，不声称已证明一般模式寻求定理。

3. Python官方math文档。
   - URL：https://docs.python.org/3/library/math.html
   - 实际打开：log、log2、log1p、fsum与isfinite条目。
   - 核对：单参数log采用自然对数，log2是二进制对数，fsum用于稳定累计，isfinite用于拒绝非有限输入。
   - 版本：在线页面当日显示Python3.14.8；作者实际运行Python3.12.14。本单元只使用这些版本中均存在的标准库接口，不把页面版本误报为运行版本。

4. Python官方decimal文档。
   - URL：https://docs.python.org/3/library/decimal.html
   - 实际打开：Decimal.ln与localcontext相关段落。
   - 核对：ln给出自然对数并按ROUND_HALF_EVEN正确舍入；localcontext用于隔离80位精度的独立算术参考。
   - 使用边界：高精度结果用于数值交叉核对，非对KL非负性的数学证明；证明在正文中完整给出。

未作为已核实正文来源的尝试：MacKay公开PDF返回403；Project Euclid的1951年KL论文页面仅返回空白可读正文；PyTorch的两个交叉熵文档地址仅返回重定向。没有据这些失败读取编造具体页码、引文或完成状态。本课运行不依赖上述站点。
