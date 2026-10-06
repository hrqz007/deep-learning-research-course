# 第057单元 语言模型实验与评价

先修022、040、055、056。此包独立包含从零训练、分组拆分、完整错误记录、NPZ检查点和可重跑实验，不需下载外部数据或模型。

- [讲义](lecture.pdf)：NLL/PPL、任务指标、分组污染检查、不确定性与盲评边界
- [实验手册](lab.pdf)：预先固定协议，训练三模型，复算全部测试记录
- [参考答案](answers.pdf)：手算、分母、区间与失败案例
- [Notebook](experiment.ipynb)：清空后顺序执行，另存notebook_replay
- [原始来源](sources.md)、[数据说明](DATA_LICENSE.md)

## 运行

在本目录执行：

```bash
conda env create -f environment.yml
conda activate dl057
python test_experiment.py
python -O test_experiment.py
python experiment.py --output replay
python make_figures.py --results replay/results.json --output replay/figures
jupyter lab experiment.ipynb
```

Python3.12与CPU即可；固定实测核心版本见requirements.txt。正式outputs含三种子全120步轨迹、原始测试记录、拆分审计、权重及空白盲评表。Notebook用新Python进程中的内核自顶向下执行通过；未测试浏览器Jupyter界面。训练每seed最多120秒，超时不自动延长预算。

图表需Noto Sans CJK，DL_CJK_FONT可指定字体文件。预生成PDF不需要安装构建工具；若重建，先安装build-tools/requirements.txt与Noto Serif CJK，再运行python build_pdf.py lecture.md，lab.md和answers.md同理。

## 验收边界

48测试文档=336目标token，但只有12独立规则前缀组。三个种子格式均正确，任务成功只有1、1、2组，单参考EM分母另为48。保留负面结果，不能以PPL低宣称事实性好。区间为固定有限合成拆分上的近似描述，未开展人工或模型裁判评价。
