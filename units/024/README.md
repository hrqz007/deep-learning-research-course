# 第024单元 逻辑回归与softmax分类

先修[023 线性训练闭环](../023/README.md)、[017 分布与似然](../017/README.md)、[019 熵交叉熵与KL](../019/README.md)、[021 可信评价](../021/README.md)。从logits、概率到负对数似然和批次参数梯度。

## 交付与顺序

- [正文11页](lecture.pdf)，9幅原创图、15题；sigmoid/softmax/CE梯度、shape、稳定性、训练与失败机制逐步推导
- [独立实验3页](lab.pdf)和[完整详解4页](answers.pdf)，含实验A至D；三份Markdown源提供
- [Notebook](experiment.ipynb)13格实际执行，3张内嵌图；[NumPy脚本](experiment.py)、[12组测试](test_experiment.py)
- data含9训练/6测试合成点、固定配置和数据说明；outputs含五份可核对产物

在单元目录运行：

```bash
python experiment.py --output outputs
python test_experiment.py
```

主脚本用Python+NumPy；测试用SciPy稳定参照与标准库Decimal。建议环境：

```bash
conda env create -f environment.yml
conda activate dl-unit-024
jupyter lab
```

Notebook重启内核并顺序运行，写notebook_outputs，再比对随附五文件。实际Python3.12.14、NumPy2.3.5、SciPy1.17.0及真实InProcessKernel。未测全新Anaconda、浏览器Jupyter、socket内核、跨平台或GPU。PyTorch只查官方接口文档，没有执行框架训练。

## 数值和协议边界

有限float64特征绝对值≤100、权重≤10000、logit≤1e6；禁止bool/NaN/无穷及错shape。非零扩展精度或CSV/JSON文本转换成0时拒绝；入口前已丢失的信息无法恢复。微小概率在计算中下溢为0仍可能发生，直接保存log概率能保留有限损失。

配置iterations、binary_iterations各0到2000，learning_rate在1e−6到5，l2在0到1；合法范围不等于每个配置都会收敛。CSV严格五列、唯一ID、train/test均非空、训练包含全部三类。所有输入检查、计算和序列化先于输出写入；输入/计算失败保留旧结果。有效运行覆盖五个同名结果，OS多文件写失败不具原子保证。

--data、--config、--output可用于另存实验。固定图和Notebook先核对默认两份输入与五份输出摘要，不会用固定解释套自定义结果。make_figures.py重建需要matplotlib与Noto Sans CJK字体；PDF重建见[共享工具](../../shared/build-tools/README.md)。复建后仍应逐页看图文。

## 关键结果

零参数三类CE为log3，梯度第一行(5/9,−5/9,0)，第二行(5/27,5/27,−10/27)，偏置行0。主实验400步、η0.25、λ0.03，训练/测试CE约0.123658/0.068982，总目标约0.238855。9/9与6/6分类正确只涉及固定合成点，不能证明真实泛化或校准。

z=(1000,1001,−1000),y=2时概率第三项下溢0，稳定CE仍约2001.313262。z=(8,0,0),y=0的正确CE约0.000670700，错误双softmax约0.551871。z=37正例的稳定二类梯度约−8.5330e−17，避免直接p−1抵消。

无正则可分二分类400步后w≈4.00124，BCE≈0.0092311，准确率早已为1；没有达到零损失的有限最优w。softmax共同平移有参数冗余；非线性概率变换不改变线性分数的类别边界。

作者检查：243个100位Decimal多类向量、300批SciPy对照、18个二类标签/分数组合、60组logit及W全坐标差分、独立手算/归约/错误链式梯度、heldout标签干预、22数学输入拒绝、22坏CSV/配置新旧输出保护、转换边界和7个绘图入口守卫；Notebook坏配置在首格拒绝。五结果跨新进程/Notebook字节同，18页PDF与9图逐一查看。详情见verification.json；独立QA和远端状态由根清单管理。
