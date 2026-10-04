# 第016单元 随机变量与条件概率

先修：[007 代数函数与图像语言](../007/README.md)。本讲回答训练数据中的随机性和条件信息应怎样表达，从有限总体建立事件、随机变量、联合、边缘、条件、Bayes与独立，再用两个原创世界区分条件和干预。

## 文件与顺序

- [正文9页](lecture.pdf)，6幅原创说明图与12题，Markdown源为lecture.md
- [独立实验3页](lab.pdf)与[详解3页](answers.pdf)，对应Markdown源随附
- [Notebook](experiment.ipynb)有11个实际执行代码格；[脚本](experiment.py)与[test_experiment.py](test_experiment.py)可独立运行
- data/inspection_counts.csv为原创1000件有限总体四格计数；data/README.md说明抽样单位
- figures、make_figures.py、environment.yml、source-checks.json、verification.json分别记录图、建议环境与核验

## 运行

核心只用Python标准库，不要求NumPy、GPU、网络或付费API。在本目录执行：

```bash
python experiment.py --output outputs
python test_experiment.py
```

需要独立Anaconda环境时，可参考：

```bash
conda env create -f environment.yml
conda activate dl-unit-016
jupyter lab
```

选择对应内核后重启并运行全部Notebook。脚本写outputs，Notebook写notebook_outputs；重跑覆盖相应目录的results.json、prevalence_scan.csv、environment.json，不改输入配置。要保留旧结果，请为脚本指定其他--output目录。仓库只保留一份脚本输出及Notebook内嵌执行结果。

可选重建图需要matplotlib、NumPy及Noto Sans CJK字体，运行python make_figures.py；这不属于核心实验依赖。PDF构建从课程根目录运行python shared/build_pdf_mathjax.py units/016/lecture.md，并对lab.md和answers.md重复，详见[构建说明](../../shared/build-tools/README.md)。任何改图需重新检查PDF。

## 关键结果

- 四格882、98、2、18；P(缺陷)=1/50，P(标记)=29/250
- P(标记|缺陷)=9/10；P(缺陷|标记)=9/58；P(缺陷|未标记)=1/442
- 固定两种标记率，先验1/1000、1/10、1/2分别给后验1/112、1/2、9/10
- 5件中2件缺陷，放回两次均缺陷4/25，不放回1/10
- 异或三个变量两两独立，却不共同独立
- 两个观测联合表相同的生成世界，do(X=1)下Y=1概率分别为1与1/2

精确Fraction计算，十进制只用于展示；模型要求四格显式给出、计数非负、总数为正、标签为整数0/1，拒绝布尔值。条件质量为0时拒绝；使用两个观察条件率的Bayes接口要求先验严格介于0和1、标记总质量正。输入比例用整数、分数字符串或Fraction，不接受二进制float。

8组测试包括100个独立编号枚举总体、25600对事件、23424次条件概率、98次Bayes核对、26项非法输入和3个畸形CSV案例，6段文档Python代码执行通过。脚本还在异目录使用python -S运行，确认核心不依赖第三方site包；Notebook新进程真实InProcessKernel顺序执行，三输出一致；15页PDF逐页目视。

未验证新装Anaconda、Jupyter浏览器界面、跨进程socket内核或所有平台。合成总体、假设参数扫描及因果玩具世界，不提供真实检查器性能、未知总体独立性或现实因果结论。作者检查见verification.json，课程独立验收与发布状态由根目录清单管理。
