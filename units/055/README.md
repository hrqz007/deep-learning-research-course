# 055 小型语言模型训练与规模

目录先修033至035、054、065；本课直接提供065所需的最小测量桥梁，无需未发布单元也可独立完成。实际训练12个CPU小decoder，记录参数、token、重复度、估算FLOPs、吞吐、RSS、验证及逐位置loss。GPU未测，字段为null。

- lecture.pdf/md：完整推导、真实两步AdamW、7幅原创图与测量边界
- lab.pdf/md：独立安装、9组练习、预算、验收和排错
- answers.pdf/md：逐步计算、全部种子与负面结果
- experiment.ipynb：已执行；Run All重训12配置并重画图
- experiment.py、generate_data.py、test_experiment.py、make_figures.py：带注释源码
- data/：256条原创唯一合成文法、128/64/64划分、词表和SHA
- outputs/protocol.json、runtime.json、results.json：固定协议、环境和全结果
- outputs/runs/：每run的80步记录及无pickle权重weights.npz

本目录运行：

```bash
python test_experiment.py
python -O test_experiment.py
python experiment.py --output replay
python make_figures.py --results replay/results.json --output replay/figures
jupyter lab experiment.ipynb
```

安装依environment.yml/requirements.txt，实测Python3.12、torch2.7.1+cpu、Linux单线程。图需Noto Sans CJK，默认/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc，可设置DL_CJK_FONT；细节见lab.pdf。每run80步、8960token，12run共107520。数值实验约一分钟但机器不同会变化；步边界180秒、进程240秒超时后不自动加预算。

权重用np.load(...,allow_pickle=False)读取，只存模型state_dict，不承诺optimizer/RNG断点恢复。测试读取随包outputs核对全部12权重，请保留。新结果写replay或notebook_replay，计时/RSS会随负载变化。

保留的负面结果：大模型在32文档池训练更低但验证更差；近似等计算预算与等token比较排序不同；确定颜色规则尚未学好。没有自然中文、长上下文、实测GPU或前沿规模外推主张。测试划分未评分。

PDF重建见build-tools/README.md。execute_notebook.py使用新进程in-process内核核验顺序执行与图片，未测试Jupyter浏览器UI或socket。数据许可见DATA_LICENSE.md，所有原始来源见sources.md。
