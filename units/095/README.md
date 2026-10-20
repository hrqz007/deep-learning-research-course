# 第095讲 流匹配与输运

原大纲G3。先阅读 lecture.md（或lecture.pdf），再按lab.md完成实验，最后核对answers.md。

## 运行

Python 3.12，NumPy与Matplotlib；全部为原创合成数据与CPU实验，无网络、GPU、付费API或模型下载。

```bash
python -m pip install -r requirements.txt
python experiment.py
python tests.py
```

Notebook为experiment.ipynb，按顺序运行全部单元格；必须先把完整单元文件夹下载到本地，不能只取Notebook。已执行输出随附，但应主动重新运行。nbclient验证的是内核执行，不声称验证浏览器Jupyter界面或所有操作系统。

## 文件

- lecture / lab / answers：Markdown和PDF
- experiment.py：详细注释的自包含实验；tests.py：数值与结构断言
- experiment.ipynb：已执行Notebook，包含真实输出与图
- outputs：真实metrics.json、data.npz及运行记录；不能将教学合成数据当真实传感器资料
- figures：三幅由实验生成的原创图
- manifest.json：文件摘要与验收记录

复现输出写到脚本所在目录outputs；重跑会替换这些可重新生成文件，修改实验时建议先复制目录。时间测量随机器负载变化，数值容差比逐字比较更合适。文献链接在讲义末尾。本讲是可解析或线性玩具机制实验，不复现论文的大规模模型结果。
