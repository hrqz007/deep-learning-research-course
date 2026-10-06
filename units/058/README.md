# 058 自编码器与潜在表示

用一个可完整复跑的反例，区分重构好、去噪好和标签容易线性读取。讲义从样本、特征、矩阵形状和MSE开始，逐步推导编码/解码、反向传播、PCA线性基准、瓶颈、去噪与稀疏约束。

## 先读和先运行什么

1. lecture.pdf / lecture.md：13页完整讲义，含手算、原创图、指标和边界。
2. lab.pdf / lab.md：5页独立实验手册，环境、逐步操作、解释和扩展。
3. experiment.ipynb：10个代码单元，已顺序执行；Run All真正重训六个网络。
4. answers.pdf / answers.md：5页逐题推导、完整数值、评价边界与评分参考。
5. experiment.py、test_experiment.py、make_figures.py：训练、数值与包完整性测试、从结果重画图。
6. outputs/：固定数据、协议、11行结果、全部6份训练权重、11份探测器、图数据。

## 环境和快速开始

Anaconda Prompt进入本目录：

```bash
conda env create -f environment.yml
conda activate dl058
python test_experiment.py
python -O test_experiment.py
python experiment.py --output replay
python make_figures.py --results replay/results.json --output replay/figures
jupyter lab experiment.ipynb
```

也可在Python3.12的venv中运行 `python -m pip install -r requirements.txt`。Jupyter应选择同一环境内核，并执行Restart Kernel and Run All。脚本与Notebook均离线生成数据，不需要GPU、下载或API。

实际验证环境：Python3.12.14、PyTorch2.14.1+cpu、NumPy2.3.5、Matplotlib3.10.8、nbformat5.11.1、ipykernel7.4.0。requirements允许兼容PyTorch2系列；这些其他版本未在本单元逐一验证，不保证跨版本/硬件逐比特相同。实际版本也记录在outputs/results.json。

图形使用Noto Sans CJK字体，默认路径为Linux的/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc。其他系统安装Google Noto官方字体后，将DL_CJK_FONT设为实际字体文件路径。纯数值训练与测试不依赖中文字体。PDF构建额外依赖与字体说明在build-tools/README.md。

## 固定协议

原创8维合成数据：640训练、160验证、320测试，固定数据种子5800。高方差干扰因素与低方差标签因素正交混合。仅用训练输入均值中心化，不逐坐标标准化。

网络8→16 ReLU→2→16 ReLU→8，共362参数，无复制跳连。AE/DAE各3初始化5801至5803，各250步、批64、Adam学习率0.005、CPU单线程。每run16000次样本曝光，全部6run共96000次。DAE额外高斯噪声标准差0.5，目标保持干净。每run最多120秒，超过即报错，不自动增加预算。

冻结表示后，只用训练前128例标签训练岭回归式线性探测，训练特征均值/标准差也仅由这128例估计，岭系数1，截距不惩罚。验证输入仅用于曲线诊断；不挑最佳检查点或种子。

## 核心结果及限制

二维AE三种子平均测试MSE为0.042547，探测准确率47.81%；DAE为0.046456与48.02%；PCA2为0.038963与47.81%。原始8维探测100%，但没有压缩。去噪MSE中DAE为0.099736，优于AE的0.113834，仍略高于PCA的0.096840。

数据是有意构造的反例，不是真实语义任务。三种子共享同一数据划分；固定线性探测失败不证明没有任何非线性可恢复信息。随机基线解码器未训练。稀疏约束只做概念、算术和测试，没有伪造稀疏AE训练结果。

## 复现与再生成

```bash
python create_notebook.py
python execute_notebook.py experiment.ipynb
python -m pip install -r build-tools/requirements.txt
python build_pdf.py lecture.md
python build_pdf.py lab.md
python build_pdf.py answers.md
```

execute_notebook.py使用新Python进程中的in-process IPython内核逐格执行并保存结果，适用于构建环境禁用socket时；这不是浏览器Jupyter界面测试。Notebook重放目录notebook_replay和脚本replay均不覆盖outputs，可自行清理。所有npz读取使用allow_pickle=False。

来源与核验见sources.md和source-checks.json，数据许可见DATA_LICENSE.md。无需其他单元文件即可运行。本单元实验到预先规定的配置结束，不继续拓展后续单元。
