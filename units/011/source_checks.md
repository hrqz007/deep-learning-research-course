# 一手来源核对

核对日期：2026-10-04。通过公开网页实际读取作者教材与官方API文档。推导、故事、图和数值核验独立编写；标准反例借助来源核对，未复制来源叙述或图像。

1. OpenStax Calculus Volume 3 第4.3节 Partial Derivatives。
   https://openstax.org/books/calculus-volume-3/pages/4-3-partial-derivatives
   核对偏导的坐标差商定义，以及其他独立坐标保持固定的含义。本讲自行选择L=x²(y+1)²，并从两条差商重新推导。
2. OpenStax 第4.4节 Tangent Planes and Linear Approximations。
   https://openstax.org/books/calculus-volume-3/pages/4-4-tangent-planes-and-linear-approximations
   核对可微余项条件、全微分，以及邻域内偏导存在且在该点连续的充分条件。本讲对主例提供独立余项界，未把有限图表当作一般证明。
3. OpenStax 第4.5节 The Chain Rule。
   https://openstax.org/books/calculus-volume-3/pages/4-5-the-chain-rule
   核对外层可微与内层可导或可微的条件，及路径导数按各条依赖贡献求和的形式。主例的48+16、32、96均独立计算。
4. OpenStax 第4.6节 Directional Derivatives and the Gradient。
   https://openstax.org/books/calculus-volume-3/pages/4-6-directional-derivatives-and-the-gradient
   核对单位方向、内积公式与非零梯度的最陡方向。页面某段将偏导存在直接连到可微，条件表述不充分；本讲采用其定理明确要求的可微假设，并用反例解释两个偏导存在不足以保证可微。
5. NumPy 2.3官方asarray文档。
   https://numpy.org/doc/2.3/reference/generated/numpy.asarray.html
   核对数组转换、dtype推断与复制行为。核心接口在转换前检查原始Python列表或元组中的布尔叶子，避免混合数值类型提升隐藏布尔值。已有数值ndarray只承诺检查其当前dtype。
6. NumPy 2.3官方isfinite文档。
   https://numpy.org/doc/2.3/reference/generated/numpy.isfinite.html
   核对逐元素有限值检测。核心使用all汇总有限性，明确拒绝NaN和无穷。函数本身不保证数值微分精度，步长和误差另行检查。

补充说明：最初还读取了NumPy stable页面；其当前页面版本为2.5，因此又核查与实际NumPy 2.3.5运行环境匹配的2.3归档页面。正文引用stable入口，实际执行语义同时经过2.3页面及代码测试核对。

实验仅展示固定CPU数值机制，不支持训练表现、一般收敛率、速度或真实设备结论。来源的许可没有自动扩展给本讲以外的内容；本讲没有附带来源全文。
