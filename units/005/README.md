# 005 数据管线与可视化检查

先修：[004 数组张量与形状思维](../004/README.md)。本讲只用四则运算建立描述、缺失填补与min-max缩放，不预设概率统计。

阅读[讲义](lecture.pdf)、[实验指南](lab.pdf)，从空会话顺序运行[Notebook](experiment.ipynb)；[练习详解](answers.pdf)包含全部手算。三份Markdown是可编辑原稿。数据字典见[data/README.md](data/README.md)。

在本目录运行：

```bash
python experiment.py
python test_experiment.py
python make_figures.py
```

最小数值实验只需Python 3.12与NumPy 2.3.5，CPU、离线、无API。Notebook另需JupyterLab/ipykernel/nbformat；图形需Matplotlib及系统Noto Sans CJK字体。environment.yml提供建议环境，未声称已完成全平台Conda安装验证。generate_data.py可重建全部原创合成CSV与名单。

正式管线处理13行原始记录、12条唯一观测，固定新设备目标的6/2/4名单，仅训练6行拟合参数。训练第二列均值38，全量错误演示为68；训练尺度[5,50]。同设备多次观测保留。主流程不评分主数据测试目标，教材公开数据并不构成真实盲测。

输出在outputs；Notebook写入notebook_outputs，包含审计、预处理参数、逐ID张量账本、固定名单、环境与数据SHA-256。独立身份记忆反例的两个MAE为0与71/3，测量不同目标；不得解读为算法优劣或现实统计结论。

make_figures.py重建7幅原创图。执行检查、来源检查、页数与限制在verification.json及source-checks.json。本文件说明作者侧产物；课程独立核验结果见相应发布记录。
