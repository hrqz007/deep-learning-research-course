# DL074 统计学习与泛化界

发行序号074明确对应原大纲 **T2**；catalog原编号及先修保持不变。原先修：[018](../018/README.md), [020](../020/README.md), [022](../022/README.md), [038](../038/README.md)。正文从基础符号重新讲起。

- [中文讲义](lecture.md) / [PDF](lecture.pdf)：连贯推导、手算、五张原创彩色机制/结果图与边界讨论。
- [独立实验](lab.md) / [PDF](lab.pdf)：环境、逐行示例、完整运行、实验设计与练习。
- [练习答案](answers.md) / [PDF](answers.pdf)：数值核验与常见误区。
- [Notebook](experiment.ipynb)：新进程真实内核按序重跑全部实验，保留输出。
- [固定实验设计](experiment_plan.json)、[数据字典](data/README.md)、[来源](sources.md)、[真实核验记录](verification.json)。

## 独立运行

在本目录运行，首次安装依赖可能联网，实际实验完全离线，仅CPU，无需PyTorch、账号、外部数据或模型API。

```bash
# 安装实验和Notebook依赖。
python -m pip install -r requirements.txt
# 正常及优化模式测试，各12项。
python -m unittest -v test_experiment.py
python -O -m unittest -v test_experiment.py
# 完整实验另存自己的目录，保留固定outputs。
python experiment.py --output my_run
```

outputs/results.json包含全部预定配置与环境；outputs/trials.npz包含未汇总数值证据和必要参数。原始合成数据随包提供，也可用generate_data.py无网络重建。默认outputs会覆盖同名文件，学习者建议使用my_run。

## Notebook、图与PDF

```bash
# 从源码生成Notebook，清除旧输出；随后务必执行。
python create_notebook.py
# 在新Python进程中的真实IPython进程内内核顺序执行。
python execute_notebook.py experiment.ipynb
# 从固定outputs重画五张图。
python make_figures.py
# 需额外文档依赖，见构建说明。
python build_pdf.py lecture.md
python build_pdf.py lab.md
python build_pdf.py answers.md
```

Notebook将完整计算写到notebook_outputs，与固定outputs分开；制作时逐数组核对重跑结果。作者回放方式是进程内内核，浏览器未测试；独立QA如验证外进程内核会在verification记录。阅读现有PDF不需构建依赖，[重建说明](build-tools/README.md)提供入口。

数据与原创材料许可见[DATA_LICENSE.md](DATA_LICENSE.md)。不把临时tmp、notebook_outputs、依赖缓存或个人日志作为发行文件。
