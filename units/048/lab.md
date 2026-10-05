# 048 视觉研究项目 实验指南

本指南可独立使用。你将从随机初始化训练一个圆盘与圆环分类CNN，比较原始输入、亮度增强、角标随机化和角标遮蔽，最后交付有数据许可、强基线、公平预算、消融、偏移检查与失败案例的研究包。所有数据原创合成，CPU离线运行，不下载预训练权重，不需要API。

先修022、040、044、045。检测/分割路线额外046，ViT额外047，预训练额外041；本次分类路线均不使用。预计阅读与手算90分钟，运行通常数分钟，讨论与报告60分钟。首次环境安装不计入；实际训练耗时依CPU而异。

## 1 建立干净环境与输入边界

进入本单元目录，先读README.md、DATA_LICENSE.md和protocol.json。已有合适的Python3.12环境可直接使用；不要把本包的依赖装进重要业务环境。conda用户可创建独立环境：

```bash
conda env create -f environment.yml
conda activate dl048-vision-project
python -c "import torch,numpy; print(torch.__version__, numpy.__version__)"
python test_experiment.py
python -O test_experiment.py
```

若只需CPU，可依据PyTorch官方安装页选择CPU wheel；requirements.txt的torch版本不强制CUDA。本包的已验证版本为torch2.7.1+cpu，完整实际版本另见outputs/environment.json。PDF已经生成，数值实验不需要WeasyPrint或MathJax。WeasyPrint与MathJax仅重排PDF时需要；但绘图和完整Notebook也必须先安装Noto Sans CJK SC，不能只靠environment.yml。当前验证Linux使用/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc，未验证其他平台全新安装。

Debian/Ubuntu可从官方系统仓库安装，在本单元目录检查：

```bash
sudo apt-get update
sudo apt-get install fonts-noto-cjk
python -c "import make_figures; print('CJK font ready')"
```

脚本优先查找SC字体，否则从上述Linux TTC提取到可写临时目录；不下载字体。其他系统需先让Matplotlib发现已安装的Noto Sans CJK SC，再执行同一检查。检查失败时先解决字体或临时目录权限，不继续绘图/完整Notebook；数值训练与测试、阅读已有PDF仍可进行。详细说明见README和build_tools/README。

允许输入为CPU float32、形状N×1×20×20、1≤N≤10000、有限且在[0,1]的像素。数据标签是int64的0或1。错误类型、尺寸、通道、NaN/Inf和越界像素应被拒绝；请勿自行把错误输入静默裁剪成“能运行”。本包不验证任意RGB照片、GPU或半精度。

**练习1 数据许可。** 写出本数据的来源、许可、是否包含自然人、可支持和不可支持的结论。若改用网络图片，需要额外核对哪四类证据？

**练习2 形状与预算。** 从N×1×20×20推导两次卷积池化与分类头的全部张量形状，手算1066参数和86,800 MAC的来历。再算24候选训练的更新总数。

## 2 数据检查与不可逆的信息边界

保留原始data目录，另生成一份，比较.npy和逐样本元数据的原字节。你可以重现测试生成过程，但不能把这一操作变成选择方法的依据。

```bash
python generate_data.py --output reproduced-data
python -c "import json; print(json.load(open('data/manifest.json'))['splits'])"
```

data有384训练、192同域开发、192偏移开发、256同域测试、256反相关测试。生成种子固定，类别各半，角标匹配目标概率依次0.90/0.90/0.50/0.90/0.10。训练增强在划分之后执行，基础对象base_id不会跨集合。manifest用于校验文件身份；逐图SHA只能排除精确重复，不能证明现实图片没有近重复。

**练习3 划分与泄漏。** 解释“先把每图增强10次，再随机分训练测试”为何有问题。设计一个最小防护检查。若使用预训练，还要加什么审计？

**练习4 理解概率。** train实际角标匹配率87.24%而非90%，test_shift为11.33%而非10%。这是否说明代码错？两个常数或角标规则的预期作用分别是什么？

## 3 不依赖自动微分的微型计算

样本A输入[0,1]、标签0；B输入[1,2]、标签1。参数(w,b,a,c)=(0.5,0.2,0.8,−0.1)。每位置先算z=wx+b、h=ReLU(z)，两位置取平均m，再算logit s=am+c、概率sigmoid(s)。批损失为两个二元交叉熵的平均，学习率0.2。

```bash
python -c "from reference import hand_ledger; import json; print(json.dumps(hand_ledger(),indent=2))"
```

程序结果应最后再看。你至少要用纸笔完成A和B前向、各自四个参数梯度、合并、同步更新以及新参数的第二次前向。

**练习5 完整前向。** 求A/B的z、h、m、logit、概率、逐样本loss与批loss。解释为什么不能把logit0.26当26%概率。

**练习6 共享路径。** 写出从批loss到logit、平均池化、ReLU、w和b的局部导数。列出两个样本、两个空间位置对w的全部四条贡献，并给出四参数总梯度。

**练习7 同步更新。** 算出四个新参数，重做A/B前向。为什么B损失可能上升而批loss下降？若先更新w再计算a梯度，优化步骤错在哪里？

**练习8 独立参照。** 正常和-O模式运行测试，解释中心有限差分在ReLU折点的局限。为什么“两个实现都调用同一个torch函数”不算强独立参照？

## 4 运行正式预写协议

首次正式结果已在outputs中保留，不可覆盖冻结目录。要重现，使用新的输出目录。重现是检验计算一致性，不能成为独立测试样本数，也不能看测试后修改协议再继续沿用相同测试作正式评价。

```bash
python experiment.py --output reproduced-run
python -O experiment.py --output reproduced-run-optimized
python make_figures.py --results reproduced-run/results.json --output reproduced-figures
```

每次运行先写protocol-before-training.json，再加载训练和两个开发集，训练24个候选。每候选18epoch×6批=108更新。固定最终epoch评价，不早停；每个方法按两开发域平均交叉熵，再跨三个种子平均选择学习率。全部方法选完后写selection-freeze.json，之后才加载测试数组，评价12个finalist。

学习率0.03和0.1；种子11/22/33；SGD动量0.9；batch64。四个方法使用同一模型、相同seed初始权重和样本顺序，增强RNG独立。确认全部24次训练完成再读图；失败候选不能默默跳过。程序若遇到非有限loss/梯度会显式失败，本轮没有此类失败。

**练习9 公平与选择。** 哪些因素被控制，哪些仍不同？mask_stamp与random_stamp是否严格单因素消融？假设某方法在测试更好却开发分数更差，应该更换“获选方案”吗？

**练习10 复现比较。** 对比正常和-O两个results.json，先各删除唯一的wall_seconds再比较全部数据。比较所有权重.npz的字节和哈希。给出“完全相同”的范围，不要声称跨硬件永远相同。

## 5 使用Notebook与保留实际输出

```bash
jupyter notebook experiment.ipynb
```

打开后重启内核并从头Run All。Notebook会在新临时目录完整重跑训练与作图，将新结果与保留版本比较，并用BytesIO和display.Image嵌入本轮生成的PNG，不能依靠旧图输出蒙混通过。

本包制作时实际使用如下无socket路线：

```bash
python build_tools/execute_notebook_inprocess.py experiment.ipynb
```

这是新Python进程中的真实ipykernel InProcessKernel，顺序执行所有代码格。它不验证浏览器Jupyter界面或外进程socket启动。若本地界面启动失败，应先记录环境和报错，再核对数值脚本是否可用；不能把未测界面写成已测通过。

验收至少包括：所有代码格execution_count连续、无error输出、训练24候选日志存在、科学结果比较为True、11个实际PNG输出、两份测试均有完整预测、所有图片清楚可读。请从本轮Notebook输出中逐张检查图例和数据是否相互遮盖。

## 6 读结果而不重新选择

outputs/results.json包含24个候选的19个时点（epoch0至18）完整训练/开发指标、每候选最终开发预测、12个finalist正式测试完整预测与反事实结果。finalist对应的.npz为最终模型参数。selection-freeze.json记录所有候选权重哈希与最终选择；它不是网络提交的时间戳注册。

**练习11 指标与误差。** 计算角标随机化相对plain的三种子配对反相关准确率差；同时比较同域准确率和交叉熵。解释为何“平均同域准确率一样”不能概括成没有任何代价。根据四组实际分母重建总体错误率，不能对四组错误率直接等权平均。

**练习12 失败与研究结论。** 取plain seed11反相关测试按索引前六个错误，写“事实—假设—下一验证”。解释只翻转角标的干预可以与不可以说明什么。用不超过150字写本轮结论，再列一个需要新测试集的后续实验。

## 7 提交清单与停下来的条件

提交研究报告应包含：问题及反驳条件；数据卡与许可；协议原文及冻结时间顺序；强基线和预算；全部候选与差种子；主指标、子群分母、跨域表现、负结果；失败案例选择规则；完整可执行代码、数据、环境与实际Notebook输出。

可以保留事后发现，但应标“探索性”，与预写分析分开。若发现数据重复、评价加载了错误权重、冻结前泄漏测试信息，应先撤回该轮正式测试结论，修复流程后重新安排真正未见的测试。不可删掉痕迹再说原结果从未有问题。

完成条件不是达到某个好看的准确率，而是同一证据链可重现且每个结论与实际数据相符。详解见answers.pdf；不要先看详解再把它当独立完成的计算。
