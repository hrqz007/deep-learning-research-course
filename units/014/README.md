# 第014单元 浮点数与稳定数值计算

先修：[004 数组张量与形状思维](../004/README.md)、[007 代数函数与图像语言](../007/README.md)、[010 导数积分与局部变化](../010/README.md)。本讲回答：数学等价的公式为什么在计算机上得到不同结果？

## 阅读与运行

- [8页正文](lecture.pdf)：二进制有效位、间隔与范围、舍入、消减、稳定softmax/log-sum-exp、差分误差权衡，6幅原创图
- [3页实验指南](lab.pdf)：七阶段手算、程序对照、失败定位与记录
- [3页详解](answers.pdf)：12道练习和各实验检查点
- [Notebook](experiment.ipynb)：10个实际执行代码格；[脚本](experiment.py)和[test_experiment.py](test_experiment.py)可独立运行
- Markdown源、data/config.json、图与生成器、environment.yml、source-checks.json和verification.json随单元提供

已有Python 3.12与NumPy环境时，在本目录执行：

```bash
python experiment.py --output outputs
python test_experiment.py
```

需要独立Anaconda环境，可按建议文件运行：

```bash
conda env create -f environment.yml
conda activate dl-unit-014
jupyter lab
```

选择此环境内核，重启后顺序运行Notebook。脚本写outputs，Notebook另写notebook_outputs；每次覆盖对应目录中的results.json、difference_scan.csv、environment.json，不改配置。要保留旧记录，脚本指定另一个--output目录。仓库含一份脚本结果和Notebook内嵌执行证据，不重复存放Notebook输出目录。

核心只依赖NumPy与标准库，不联网、不用GPU或付费API。可选图重建需matplotlib与Noto Sans CJK字体，运行python make_figures.py。PDF重建从课程根目录运行python shared/build_pdf_mathjax.py units/014/lecture.md，并对lab.md与answers.md重复操作；参考[构建工具说明](../../shared/build-tools/README.md)。图改动后须重新渲染目视PDF。

## 结果与检查

- (1000,1001,999)朴素softmax非有限；稳定权重约(0.244728,0.665241,0.090031)
- LSE约1001.4076059644444；float32尾概率下溢为0时，直接log权重仍可约为-200
- (100000000,100000001)转float32丢失差1，稳定输出(0.5,0.5)；float64约(0.268941,0.731059)
- float64中sqrt(1+1e-16)-1直接式为0，有理化式约5e-17
- 两dtype共26个中心差分步长记录，拒绝行保留空误差而非伪造零值
- 9组测试、100向量×2dtype的110位Decimal参考、82个导数参考、9项Fraction恒等式、23拒绝案例及6段文档代码
- 新进程脚本和10格真实InProcessKernel执行，三个输出逐字节一致，14页PDF逐页检查

稳定接口只接受非空一维有限实数分数与float32/float64；检查转换后及平移差、求和、输出的有限性，极端差溢出时拒绝。布尔值、复数、文本、非有限值和错误shape拒绝；已有数值数组此前丢失的类型来源无法恢复。朴素接口是故意保留的失败基线。

没有新装Anaconda、浏览器Jupyter、跨进程socket内核或其他硬件验证。参考计算是高精度有限计算，不是无限精度真值；本课不提供真实训练、低精度加速或所有动态范围准确性保证。独立课程验收与发布状态由根目录清单管理。
