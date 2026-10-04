# 031 梯度核验与最小调试法

核心问题：训练失败时先检查哪一层证据最省时间？先修：[027](../027/README.md)、[030](../030/README.md)。本讲独立运行，不导入其他单元。

## 阅读顺序

1. [正文讲义](lecture.pdf)：同一4样本、13参数的完整前向、局部导数、链式累计、全部参数梯度、SGD更新、新前向，再做有限差分。不是只给最终矩阵公式。
2. [独立实验指南](lab.pdf)：从数据契约到可定位故障报告。
3. [已执行Notebook](experiment.ipynb)：13个代码格，全部trace坐标与3幅图，干净内核顺序运行。
4. [完整答案](answers.pdf)与[可搜索答案源文](answers.md)。
5. [来源与实际查阅范围](sources.md)、[验证记录](verification.json)。

## 文件与实际结果

- `experiment.py`：实际CPU PyTorch程序。`test_experiment.py`：16组测试，标准库、NumPy、PyTorch。
- `data/batch.csv`：8条合成样本、规则标签与固定随机回归标签；构造见[data说明](data/README.md)。
- `data/config.json`：固定配置。`data/figure-input-sha256.json`：2输入+7参考结果摘要。
- `outputs/full-trace.json`：同一13参数例的完整张量、shape、4×13样本贡献和77个坐标对照。
- `outputs/finite-difference.csv`：13坐标×6步长共78条；`training.csv`：两次拟合各1601行。
- `outputs/faults.json`：正常、停步、零学习率、漏参、断图、归约、不可导点、错误标签仍通过等对照。
- `outputs/perturbations.json`、`fitted-values.csv`、`summary.json`：干预、逐点拟合值与摘要。
- `make_figures.py`与`figures/`：9幅原创彩色机制/结果图。

固定13参数初始loss为0.07877855276232809，77坐标核对最大差约5.55e-17，h=1e-6的差分最大误差约1.21e-11。一次SGD学习率0.1后，重算loss为0.07241983164894575。两次49参数拟合都达到half-MSE < 1e-4；随机标签也能记住，不能由此推断泛化。模型拟合原8点极好，小幅输入扰动的平均变化率仍约0.5838，区别于生成规则0.7。

## 环境与运行

实际验证：Linux x86_64、Python3.12.14、NumPy2.3.5、PyTorch2.7.1+cpu、单线程、float64。CPU即可，无显卡、无数据下载、无模型下载、无付费API。没有在用户机器执行安装。可以使用已有兼容环境，也可自行建立隔离环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.3.5
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install jupyterlab ipykernel nbformat matplotlib==3.10.8
```

Windows激活方式不同，可使用`.venv\Scripts\activate`。这些是学习者安装说明，不表示所有平台均已实测。[environment.yml](environment.yml)提供Anaconda环境声明；全新conda安装未重新测试。先确认选择的Notebook内核就是装有这些依赖的环境。

在本目录执行：

```bash
python experiment.py --output outputs
python -m unittest -v test_experiment.py
python -O experiment.py --output optimized_outputs
jupyter lab experiment.ipynb
```

脚本从自身路径定位data，与终端当前目录无关。Notebook请在本讲目录打开并重启内核后运行全部。交付版实际使用全新Python进程中的IPython InProcessKernel执行，所有13格成功；浏览器UI和socket传输未测试。

Notebook重算到`notebook_outputs/`，7份文件与参考输出逐字节比较。该目录是本地产物，不属于必须下载的课程源文件。跨版本或平台不承诺逐字节一致；若数值只差尾数，先独立核对公式、容差与版本，不要改摘要伪装教材图未变。

## 保护参考结果与自由探索

固定CLI在任何计算和输出写入之前检查batch.csv、config.json的字节摘要。因此修改空白、标签、参数、步数或JSON超小数字都会明确拒绝。计算、校验和序列化全部成功后才建立目标目录或替换文件；每文件替换具原子性，不声称多文件的断电事务保证。

固定图像脚本和Notebook第一格核对2输入及全部7结果后才导入绘图/训练功能或写输出，防止新数据与旧图混用。这不是通用数据加载器。想改变数据或训练设置，请调用`validate_batch`、`full_trace`、`central_difference`、`train_one_batch`、`one_step_probe`等公开函数，并在新文件/目录中标注新实验，不覆盖参考图与结论。

`validate_batch`要求X为(n,2)、y为(n,1)、1≤n≤64、有限非布尔实数且绝对值≤1000。13参数助手要求theta为(13,)；差分step为[1e-12,0.1]的Python实数；训练步数1–5000，学习率(0,0.1]，seed为[0,2^31)的整数。shape检查发生在广播运算前；有限差分还要求加减扰动后的参数留在支持的数值范围内。`flat_loss`和`Tiny.forward`是内部张量表达式，调用者需保证契约；它们不重复承担完整的外部输入验证。

重建中文图需要本地Noto Sans CJK字体。`make_figures.py`默认寻找Linux常用字体位置，其他系统应配置对应字体；计算核心和已提供PDF/PNG不依赖字体安装。

## 能说明与不能说明的事

可以复核当前函数的梯度、图连接、SGD参数更新、单批次可拟合性和有限输入干预。没有独立验证/测试集，没有真实任务基准，没有GPU/分布式/混合精度测试，未实现面向任意网络的自动诊断器。`train`与`eval`在本无Dropout/BatchNorm网络中没有数值差异。正确性检查不能替代统计评价。
