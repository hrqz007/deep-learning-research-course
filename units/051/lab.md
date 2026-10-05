# 051 实验指导：门控、状态与真实自由生成

先完成A至D的数值题，再运行主实验。最终提交不仅要有loss曲线，还要有每个时间步的梯度来源、生成序列及停止原因。

<style>table{break-inside:avoid}.math-display img{display:block;margin:0 auto;vertical-align:baseline!important}</style>

## 1 环境和运行顺序

沿用050的独立Python环境。固定验证环境为Python3.12、NumPy2.3.5、PyTorch2.7.1 CPU、Matplotlib3.10.8；Notebook还需nbformat5.11.1和ipykernel7.4.0。数据本地生成，不需下载或API密钥。完整环境见environment.yml和runtime_versions.json。

从课程根目录执行；新建环境可用前两行，已安装者跳过它们：

```bash
conda env create -f units/051/environment.yml
conda activate dl051
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python units/051/test_experiment.py
python -O units/051/test_experiment.py
python units/051/experiment.py --output tmp/051-my-run
python -O units/051/experiment.py --output tmp/051-my-run-O
python units/051/audit_records.py --output tmp/051-my-run
jupyter lab units/051/experiment.ipynb
```

已有outputs用于独立重载测试，不要先删除。新实验写到临时目录，以便和原件比较。audit_records.py逐一用NumPy反传重算全部1800个已保留更新，含裁剪范数和因子，不只检查终点。Notebook Restart Kernel→Run All会重训全部六模型；新结果临时保存后与保留文件核对。

绘图和完整Notebook都需要Noto Sans CJK字体，不能只安装environment.yml就假定字体存在。已验证Linux字体路径/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc；Debian/Ubuntu可从官方仓库安装fonts-noto-cjk。其他系统将DL_CJK_FONT设为本机实际Noto CJK的ttc/otf路径，不声称已验证所有平台。安装后在单元目录执行以下检查，失败则先修字体，数值脚本和已生成PDF仍可单独使用：

```bash
python -c "import make_figures; print('CJK font ready')"
```

普通数值运行不需要PDF重建依赖；重建见shared/build-tools。浏览器Jupyter UI和外进程socket不在离线新核复验范围，跨平台安装和浮点逐位一致也不是本课已证明的范围。

## 2 A：从方程画GRU与LSTM

对GRU逐项标出x、旧h、r、z、q、n、新h，写出维度；用颜色区分保留直路、候选路径和门参数路径。确定本课z究竟表示保留还是写入比例。列出reset-after与reset-before候选的两个公式，用一个H=2的非对角循环矩阵构造二者不等的数值例。

对LSTM标出旧c、遗忘、写入、新c、输出门和h。计算c_prev=1.2、f=.8、i=.25、g=-.5、o=.6时的新c和h。先固定门求直路导数，再解释为什么真实循环模型的完整导数不能只保留这一个数。

验收：GRU保留门系数与官方PyTorch2.7 GRUCell一致；LSTM两种状态不混淆；参数公式明确是否两份bias、是否projection/双向/多层。

## 3 B：完整两样本手算

使用讲义14个标量参数、两个输入[1,-1]和[.5]，目标.2和-.1，各自h₀=0。只在最后状态用输出头预测，L为两条half-MSE的平均。

1. 逐时间计算每个仿射、r/z/n和h。写出两个最终预测、残差、单样本loss与平均loss。
2. 从loss开始，先加入批次平均1/2，再乘输出权重得到最后h上游。
3. 沿乘加、tanh、sigmoid分解局部导数，分别算输入侧/隐藏侧仿射上游。特别保留候选隐藏偏置上游a_n'×r。
4. 按时间反传，记录旧h的直路、其他路径、总上游。对每个参数做逐时间贡献表，再跨样本相加。
5. 学习率.2同步更新所有14参数，重置h₀并重算全部前向。继续第二轮梯度、更新和第三次前向。

用hand_calculation.py的Decimal70前向作独立高精度对照；用每个参数±1e-6中心差分核对14梯度。再用autograd检查三个原输入位置的梯度，而不只是参数梯度。误差需要给出绝对量级与容差理由。

验收：平均loss依次约.0390346214、.0360154925、.0338837660；样本2却变差，必须保留并解释。第二轮仍要有每个共享参数的来源，不能只贴新loss。

## 4 C：基础算子实现cell与完整BPTT

先实现单步GRU，然后把同一组权重复制到官方GRUCell。随机B=4、D=2、H=3，同时检查输出、对输入x、对旧h、全部权重与两份bias的梯度。不能只在h₀=0时检查循环权重，那会漏掉许多路径。

再搭D=H=2的小型encoder-decoder，V=8，输入包含重复符号及不同长度。NumPy手写每一步缓存，decoder反向所得最终context梯度作为encoder反向起点；共享E同时累加encoder和decoder读取路径。RNN与GRU都要对照torch；逐坐标有限差分时跳过固定PAD行，其余全部检查。

检查全局clip：先得到所有参数梯度，计算一个共同L2范数和一个共同缩放因子，再更新。把clip阈值改大可作为局部单步诊断，但不要据此改写主实验的固定协议。

验收：小RNN共64参数坐标，其中2个PAD坐标固定；小GRU112坐标，其中2个固定。分别完成62和110个自由坐标的FD；shape、mask和共享E累加均正确。

## 5 D：边界、padding与因果关系

对内容[A,B]、[C]、空串，手写encoder输入、decoder训练输入、目标和各自mask。分别把补齐宽度再加5，要求最终encoder context、有效loss、全部梯度不变。故意只mask loss而不冻结encoder padding状态，观察为什么原文终态会改变。

固定同一个encoder context与同一个decoder前缀，只修改一个较晚的decoder输入token，验证此前logit保持不变。这是decoder时间因果性的测试。不要改源序列来测试这个性质：encoder本来就允许读取完整源输入，源后部改变可合法地改变第一个输出。

让输出头恒偏向EOS，检查立即终止；再恒偏向A，检查达到max_steps时保留未结束；恒偏向PAD，检查无效符号被记为错误，没有偷偷过滤。EOS之后的状态冻结只为已结束样本，不能让它提前终止别的样本。

验收：空内容仍有BOS→EOS目标；预测不使用真实目标长度停止；外部普通内容不能包含控制ID；过长、非法ID和非有限参数明确报错。

## 6 E：teacher forcing指标与自由输出

对一个固定模型，分别打印teacher-forced逐位置argmax，以及从BOS开始逐步喂回自己输出的自由序列。报告NLL、teacher token准确率、自由金标位置准确率、整序列exact、EOS率、非法控制符率。

证明在讲义列出的确定性greedy条件下，teacher-forced“每个位置都正确”与自由exact等价。然后说明为什么这一命题并不使自由生成多余：它仍给出错误后走向、长度、非法符号与实际输出内容，也不让token准确率自动成为序列准确率。

找出RNN种子5113自由位置准确率略高于teacher token准确率的真实结果，解释为何错误前缀后的偶然修正并不违反上述exact等价。不要把这个单次现象推广成自由生成更可靠。

## 7 F：完整固定六模型和长度外推

固定数据拆分：train96、validation32、test64，内容长度2..4；long_test64，长度5..6。精确源序列不重叠。两模型各三个种子，300次全批次SGD，η=.5、全局clip阈值1，无验证选模。

保留全部1806参数状态与1800次范数/裁剪记录。独立从最终权重重算两个测试集的全部logit、NLL、自由输出和错误列表。核对普通RNN696、GRU1752坐标，不用“同H”冒充等参数或等FLOPs。

原样保留以下结果：本预算GRU没有赢过RNN；两种模型长序列exact全0；有RNN长序列未输出EOS；置零encoder context后仍可能碰巧成功一个样本。置零是预先定义的干预诊断，不是重新训练的公平容量对照。

验收：所有种子与失败都在，test未参与任何选择，初始协议文件/数据SHA不变。不要因为结果不理想追加训练并替换原记录；新增实验应先单独写协议。

## 8 G：科研小结

写一页小结：当前证据支持什么、不支持什么？区分门控数学能力、当前优化结果和长度泛化。提出一个新可证伪对照，例如等参数RNN/GRU、不同训练长度范围或加入注意力，但一次只改一个明确因素，并说明新的预算、拆分与选择规则。

提交完整手算账本、两模型cell参照与FD、padding和EOS失败探针、六模型复现、全生成列表和科研小结。Notebook需有连续执行计数和实际PNG；只截图最终loss不构成完整实验记录。
