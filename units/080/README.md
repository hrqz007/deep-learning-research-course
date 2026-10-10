# DL080 微分方程与可微模拟基础

发行序号080对应原catalog **S3**。原先修 010, 011, 015, 029 保留，不修改大纲依赖。

## 完整自学材料

- [正文PDF](lecture.pdf) / [Markdown](lecture.md)：逐步解释、推导与六张原创图
- [独立实验PDF](lab.pdf) / [Markdown](lab.md)：命令、纸笔题、故障检查与研究任务
- [完整答案PDF](answers.pdf) / [Markdown](answers.md)：推导过程与真实结果
- [已执行Notebook](experiment.ipynb)、[源码](experiment.py)、[测试](test_experiment.py)
- [研究计划](research_plan.json)、[数据字典](data_dictionary.md)、[来源](sources.md)、[核验记录](verification.json)

## 独立运行

CPU即可，安装后完全离线，无账户、GPU、外部模型和付费API。在本目录执行：

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_experiment.py
python -O -m unittest -v test_experiment.py
python experiment.py --output my_run
python create_notebook.py
python execute_notebook.py experiment.ipynb
python make_figures.py
```

两种模式各预期15项测试。Notebook在新Python进程的真实进程内内核顺序执行，另存notebook_outputs，不声称验证Jupyter浏览器UI。课程参考结果在outputs，NPZ仅存数值，可用allow_pickle=False读取。图示默认读取outputs。环境实测版本、PDF页数与校验值在verification.json；可选PDF重建见build-tools/README.md。不要发布tmp、__pycache__、.venv和重复notebook_outputs。
