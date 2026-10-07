# DL061 对抗生成与博弈训练

判别器提供的学习信号怎样改变生成器分布？本单元从二元分类损失、理想判别器和JS联系，走到可重跑的二维GAN和一次真实训练坍塌。

## 阅读顺序

1. lecture.pdf / lecture.md：零基础衔接、完整推导、图解和实测解释。
2. lab.pdf / lab.md：环境、更新边界检查、三种子实验、坍塌诊断和研究设计。
3. answers.pdf / answers.md：逐步计算、详细诊断与结论边界。
4. experiment.ipynb：已从新IPython进程顺序执行，包含真实重新训练与输出。

目录先修：019、032、030。训练不使用外部数据、预训练模型、付费服务或GPU。

## 快速重跑

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_experiment.py
python -O -m unittest -v test_experiment.py
python experiment.py
python make_figures.py
python create_notebook.py
python execute_notebook.py experiment.ipynb
```

推荐独立Python 3.11/3.12环境。制作版本见verification.json。PyTorch可通过官方CPU安装渠道安装；环境文件不固定操作系统级依赖。完整训练为分钟量级，耗时依设备而变。同名实验输出将覆盖，保留扩展研究请使用--output。

## 结果摘要与边界

基准种子11、23、37各3000轮，均覆盖8/8模式；有效比例分别0.6243、0.6748、0.6042。有效比例并非100%，不能称为完全分布匹配。

压力种子17使用公开的失衡优化配置：G学习率0.002、D学习率0.0001，每轮5次G更新。2200轮轨迹中，第800轮实际出现1/8模式且有效比例100%的单团输出；最终2200轮变成偏离所有真实团的集中输出，有效比例0。诊断快照与最终模型分开保存和解释。人为常数负对照只用于验证评价器，绝不作为真实训练坍塌证据。

## 文件说明

- experiment.py：数据、G/D、独立更新、指标与全部训练。
- test_experiment.py：13个行为与数学测试，普通和-O模式均运行。
- make_figures.py、figures/：从真实输出生成的5张原创图。
- data/mixture.npz：固定训练与独立参考数据、标签和中心。
- outputs/results.json：全量配置、指标、每100轮轨迹与诊断选择。
- outputs/*.pt：三个基准、压力最终G/D和压力诊断G权重；只加载信任来源的文件。
- outputs/*_samples.npz：各记录时点的4096个固定噪声生成点。
- create_notebook.py、execute_notebook.py：创建并顺序执行Notebook。
- build_pdf.py、build-tools/rebuild.sh：文档构建与重建入口。
- source-checks.json、sources.md：已核验的一手来源及适用范围。
- verification.json：实际测试、Notebook、PDF与数值检查摘要。
- DATA_LICENSE.md：原创合成数据和素材的来源、授权与限制。

Notebook重跑输出写入notebook_outputs，避免覆盖教材outputs；这些重复训练缓存不在固定交付清单内。PDF构建的tmp和图像审查中间文件也不在交付清单内。

## PDF重建

```bash
python build_pdf.py lecture.md
python build_pdf.py lab.md
python build_pdf.py answers.md
```

需要Noto Serif CJK SC / Noto Sans CJK SC字体、WeasyPrint系统库；Poppler用于独立渲染审查。公式本地转为SVG，不依赖在线数学渲染服务。顺序Notebook执行验证使用无socket的IPython内核，不声称已测试浏览器Jupyter界面。

所有来源仅支持概念和API；本包的训练结果来自本次独立实验。无GitHub写入、无外部数据上传、无科研论文成绩复现声明。
