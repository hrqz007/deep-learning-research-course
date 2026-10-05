# 049 实验指导：从原始文字到可核对的梯度

目标是独立重建tokenizer、padding、嵌入反向和六次固定训练，而不是只运行一个API。建议先完成A至D，再打开答案。每题都要留下输入、预测值、实际结果和解释。

<style>table { break-inside: avoid; }</style>

## 1 环境与文件

需要Python3.12、NumPy2.3.5、PyTorch2.7.1、Matplotlib3.10.8、nbformat5.11.1和ipykernel7.4.0；完整固定环境见environment.yml与runtime_versions.json。普通学习运行不需要PDF构建依赖。数据完全由本地generate_data.py生成，不下载文本，不需要API密钥。

在课程根目录建立独立环境，按environment.yml安装依赖。官方PyTorch CPU轮子可用官方索引安装torch==2.7.1；如用现成环境，先打印实际版本，不要假定任意版本或GPU都逐位相同。运行前限制CPU线程，命令如下：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python units/049/test_experiment.py
python -O units/049/test_experiment.py
python units/049/experiment.py --output tmp/049-my-run
python -O units/049/experiment.py --output tmp/049-my-run-O
python units/049/make_figures.py --output tmp/049-my-figures
```

test_experiment.py会验证已保留的六个最终模型；先不要删除outputs。要检查自己重训的最终模型，可将实验目录复制到新的临时工作目录，再把新结果放入副本outputs并运行测试，不覆盖原件。三个输出文件之间由results.json中的SHA相互关联。

Notebook从其所在units/049目录运行，或者从课程根启动后由首格定位。Restart Kernel→Run All应该真正完成六次训练，而不是只读取保存的图。新结果写入临时目录，随后与保留结果比较，结束后临时目录释放。Notebook的图由新核计算/绘制并实际嵌入PNG。若在无socket的构建环境，可从单元目录用共享execute_notebook_inprocess.py；它使用新的Python进程和真实IPython核，但不代表浏览器Jupyter界面已经测过。

## 2 练习A：逐层计数与可逆性

1. 不运行代码，写出“材料A”、预组合café、分解形式cafe加U+0301、女科学家ZWJ序列、字面量“&lt;PAD&gt;”和空串的码点数、UTF-8字节数。
2. 用text.encode('utf-8')检查字节hex。解释为什么一个字节ID未必能单独解码成Unicode字符。
3. 构造ByteBPE()，对每条探针编码，加上边界后再解码。验证decode(encode(text))精确等于原文；不要偷偷调用normalize。
4. 查看训练集码点词表的UNK次数，解释为什么它与字节词表的覆盖范围不同。未知字符变成多个字节token，不意味着模型已学会未知词。
5. 手工输入[BOS,259,EOS]，记录严格decode的失败。这个ID属于词表，为什么仍不是合法UTF-8文本？

验收：原文、码点、hex、内容ID、边界ID分列；空串被编码为[BOS,EOS]。对乱码失败保留异常类型与解释，不用replace解码隐藏问题。

## 3 练习B：从计数实现BPE

先只用两篇“abab”，预算2轮。写出初始ID、每种相邻对频次、选中对、新ID、替换后序列。再用“aaaa”预算4轮，区分重叠计数和非重叠替换。最后用两篇独立文档“a”“b”，验证不能跨文档生成ab。

实现一个不用ByteBPE.fit的标量版本：每轮逐文档逐位置计数；按(-频次,左ID,右ID)排序；用while指针替换，匹配成功前进2，否则前进1。检查平局“abac”“abac”先选哪个对。把所得合并表交给ByteBPE构造器，要求对新文字产生同一编码。

在真实训练语料上学习16轮，检查总内容token1458→1050、V260→276。查看new_piece_hex，找出至少一个不能单独UTF-8解码的合并片段。说明我们允许空格参与合并，也没有词尾标记；因此结果不等于经典按词BPE或现成tokenizer输出。

验收：同一训练集与固定平局规则产生完全相同合并表；validation/test不进入fit。额外实验若改合并预算，要先写下新协议，不把看到测试结果后的最佳预算冒充原实验。

## 4 练习C：目标位移、补齐与损失分母

给三条内容[A,A,B]、[B]、空串，统一ID为A4、B5、PAD0、BOS1、EOS2。补齐宽度取5，手写3×5输入x、目标y和mask，并给出每条有效长度与总M。

将相同序列分别补齐到5和9，生成固定的随机E/W/b，要求有效平均NLL和全部梯度不变。用独立NumPy公式与torch交叉熵比对。故意做两个错误版本：把PAD算入loss，或让池化分母使用矩形宽；指出各自改变了哪个数学目标，再恢复正确实现。

思考：忽略PAD目标是否把输出logit的PAD类别从softmax中移除了？把E[PAD]置零是否足以消除输出偏置造成的padding损失？EOS是有效目标，BOS是否是主实验里的目标？

验收：扩展右padding不改变科学结果；空内容文档仍有BOS→EOS；不产生跨文档训练对。每条序列loss平均再平均与全部有效token统一平均的差别要用一个长短不同的反例说明。

## 5 练习D：同一实例完成两次SGD

使用讲义的六行标量嵌入E=[0,.2,.4,.3,.5,-.25]，w=2、b=.1，两个内容样本[A,A,B]、[B]，目标1和0。不要调用autograd，先在纸上完成：

1. 逐位置查表、有效长度、均值、预测、残差、half-MSE、批次平均loss。
2. 写出prediction→mean→每个位置的局部导数；在每个路径最前面加入批次平均的1/2。
3. 把A两个位置累加、B跨样本累加，再算w/b。解释为什么第一次w总梯度为零。
4. 用η=.1同步更新全部参数，重新查表完成下一次前向。不要用旧均值。
5. 继续第二轮反向、同步更新、第三次前向，记录两个单样本损失和平均loss。

用Fraction做独立有理数参照，再与outputs/results.json中的hand和torch图对比。固定PAD行不参与有限差分；没有读取的特殊行应为零数据梯度。

验收：两次更新后的平均loss约为0.036908642和0.017393879；还必须给出每个样本、每个共享参数的来源，单有最终两个数字不算完成。

## 6 练习E：独立推导完整分类反向

小词表V=7、D=2，用两条内容[4,4,5]和[6]。随机参数必须固定种子；所有实数参数float64。手写稳定softmax及有效交叉熵，再按以下顺序推导并实现：

- dz = mask × (概率 - 目标one-hot) / M
- gW = 所有位置外积之和，gb = dz之和
- dh = dz × W转置
- gE = 按输入ID scatter-add，固定PAD行归零

用每个非固定参数±1e-6中心差分复核，并与torch的真实计算图核对。额外检查同一ID重排的等价性：重排E行、W列、b坐标以及输入和目标后，loss应不变，梯度应按相同规则重排。不要只重排输入ID。

验收：分别报告最大loss差、三组梯度差和FD差；容差与float64及步长匹配。极端溢出应明确拒绝，不能改成静默裁剪后仍声称是原目标。

## 7 练习F：重跑固定实验，保留失败

重跑两种tokenizer×三个种子，共六个模型，每个120次全批次SGD。核查训练目标数1506/1098；保存全部121个参数状态和最终模型。不要因unigram更好而增加训练轮数、删除失败种子或用测试选checkpoint。

从final_states.json.gz独立恢复E/W/b，逐文档逐位置累加测试NLL，核对每篇和总体值；不能只调用训练时同一个metrics函数自证。重新计算token平均NLL、单模型token perplexity和bits/UTF-8 byte。三个种子结果全部展示，再取平均。

说明为何纯字节平均PPL约91.714，BPE约126.141，但字节口径分别约6.7333和5.2962。再计算加一平滑unigram基线，解释神经模型为什么不能被写成“取得更优结果”。这次实验只能证明这个固定协议下的表现，不能证明更长训练也无效。

## 8 练习G：研究审计与扩展提案

提交一个两页以内研究记录：问题、可证伪假设、数据来源和拆分单位、tokenizer版本/合并表、控制符、mask/分母、模型/预算、全种子结果、负面结果、限制。扩展提案任选一个，但先写协议再运行：

1. 比较固定更新数与固定有效token预算，另外报告参数量与计算量。
2. 使用NFC规范化，分别报告字节可逆性改变、码点/token计数与明确的评价单位。
3. 保持tokenizer冻结，加入050的RNN状态；规定独立文档重置及padding处理。

扩展结果与本次基线分开存放。不要用本课的极小模板文本发表“某种tokenizer普遍优越”的结论。

## 9 提交清单

提交标量BPE、三矩阵padding表、两轮精确SGD账本、独立NumPy/FD结果、六模型重跑与每篇NLL、所有负面结果和研究记录。环境、数据SHA、合并表和结果文件SHA一并保留。Notebook需有连续执行计数和实际图像输出，不能只有源码或预先粘贴的截图。
