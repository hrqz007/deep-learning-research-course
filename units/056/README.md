# 第056单元 生成解码与KV缓存

先修054。固定模型的训练背景见055，但本包包含所需权重、词表与代码，可独立运行。CPU、离线、无付费API。

- [讲义](lecture.pdf)：logit、温度、top-k/top-p、停止规则、逐层缓存、位置与非方形mask、计算和内存边界
- [独立实验](lab.pdf)：手算、160次采样、完整路径对照与公平计时
- [练习详解](answers.pdf)：逐步计算、shape、反例及负面结果
- [Notebook](experiment.ipynb)：可清空后从头顺序运行；输出保存在notebook_replay
- [来源](sources.md)、[数据说明](DATA_LICENSE.md)

## 运行

在本单元目录执行：

```bash
conda env create -f environment.yml
conda activate dl056
python test_experiment.py
python -O test_experiment.py
python experiment.py --output replay
python make_figures.py --results replay/results.json --output replay/figures
jupyter lab experiment.ipynb
```

亦可在Python3.12环境用python -m pip install -r requirements.txt。核心版本按本次CPU实测固定，安装PyTorch可使用其官方CPU索引；GPU并非要求。Notebook由新进程中的ipykernel顺序执行通过；浏览器Jupyter交互界面不是本次验收范围。

## 文件与验收

experiment.py负责概率规则、生成与计时，model.py明确实现完整及缓存前向，test_experiment.py在普通和-O模式均保留检查。data/含原创合成数据、NPZ权重和来源SHA；outputs/是本次完整结果，重跑不会默认覆盖它，除非主动使用默认输出目录。

验收时检查160条采样、缓存logit最大绝对差小于2e-5、停止边界与B2分块测试通过。采样计数依赖固定软件/设备；计时随负载变化，不作为逐位复现门槛。仅7输入位置，不能推断自然语言质量或生产系统提速。

## 重建PDF

预生成PDF可直接阅读。重建需要build-tools/requirements.txt及Noto Sans/Serif CJK字体。图表字体可通过DL_CJK_FONT指定。

```bash
python -m pip install -r build-tools/requirements.txt
python build_pdf.py lecture.md
python build_pdf.py lab.md
python build_pdf.py answers.md
```
