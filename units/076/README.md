# DL076 贝叶斯深度学习与模型集合

发行序号 **076** 对应原catalog **T4**，保留原先修 017, 019, 060, 071；没有修改catalog编号或依赖。三种方法在远域均发生零覆盖；工作模型后验精确不代表真实世界可靠。

## 完整自学材料

- [讲义PDF](lecture.pdf) / [Markdown](lecture.md)：逐步推导、六张原创彩色图、实测反例与边界。
- [独立实验PDF](lab.pdf) / [Markdown](lab.md)：逐行命令说明、纸笔题、故障检查与报告任务。
- [完整答案PDF](answers.pdf) / [Markdown](answers.md)：计算过程、真实结果和不过度外推的报告范例。
- [已执行Notebook](experiment.ipynb)、[完整源码](experiment.py)、[测试](test_experiment.py)。
- [实验设计](research_plan.json)、[数据字典](data_dictionary.md)、[官方来源](sources.md)、[核验记录](verification.json)。

## 独立运行

在本单元目录执行；安装后完全离线，无账号、GPU、PyTorch或外部数据。固定参考结果保存在outputs，自己的实验建议另存my_run。

```bash
# 首次安装实验依赖，可能访问软件包注册服务。
python -m pip install -r requirements.txt
# 验证数值数学与输入边界，预期7项。
python -m unittest -v test_experiment.py
# 优化模式仍执行相同检查。
python -O -m unittest -v test_experiment.py
# 真实训练并另存所有结果。
python experiment.py --output my_run
```

[environment.yml](environment.yml)提供可选conda环境；实际验证版本记录于verification.json，不声称所有操作系统都已安装验证。outputs包含data.npz、weights.npz、predictions.npz和results.json，均由源码实际计算。权重使用普通数字NPZ，可用allow_pickle=False读取。

```bash
# 重建Notebook结构（清空旧输出），再从新内核顺序执行。
python create_notebook.py
python execute_notebook.py experiment.ipynb
# 读取固定证据重建图像。
python make_figures.py
```

Notebook另写notebook_outputs，完整重训。作者执行器为真实IPython进程内内核，浏览器UI不在验证范围。六图与PDF均已进行逐页/逐图渲染审查；独立QA结果以课程发布记录为准。可选PDF重建见[build-tools/README.md](build-tools/README.md)。不要公开tmp、依赖缓存或重复Notebook输出。

本讲的结果只针对明确合成世界；方法和因果范围见正文。原创授权与来源范围见[DATA_LICENSE.md](DATA_LICENSE.md)。
