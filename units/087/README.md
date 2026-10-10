# DL087 Agent与检索系统的研究评价

对应既定大纲L5。零基础桥接、原创机制实验与研究边界。

## 阅读顺序
1. [讲义PDF](lecture.pdf) / [源码](lecture.md)
2. [实验PDF](lab.pdf) / [源码](lab.md)
3. [习题答案PDF](answers.pdf) / [源码](answers.md)
4. [执行过的Notebook](experiment.ipynb)、[实验源码](experiment.py)、[测试](test_experiment.py)、[来源](sources.md)、[数据字典](data_dictionary.md)

## 运行
建议Python 3.12虚拟环境，先安装requirements.txt。随后运行 python experiment.py 与 python -m unittest -v test_experiment。核心实验仅需NumPy（086仅标准库），数据随代码生成；安装后全程离线，无GPU、账号或付费API。

PDF重建：python build_pdf.py lecture.md；对lab.md、answers.md同理。需要系统Noto CJK中文字体。图由make_figures.py重建。Notebook在全新进程中的真实IPython InProcessKernel顺序执行；未测试网络传输或浏览器UI。完整实验数值见[outputs/metrics.json](outputs/metrics.json)，可用随课测试与重放脚本核验。

所有实验均为明确限定的教学机制或mock系统，不能当作真实LLM性能或安全结论。
