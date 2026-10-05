# 046 检测分割与结构化预测

先修039、044、045。建议顺序：lecture.pdf → lab.pdf → experiment.ipynb；独立完成后核对answers.pdf。三份文档均保留Markdown源。

本讲从整图标签讲到框和像素，逐段推导IoU、框编码/Smooth L1、像素BCE与Dice，区分训练匹配、NMS、评估匹配与AP。真实主实验为原创16×16二值语义分割，包括空图、重叠与分离目标。21次固定训练及50个阈值候选全部保留，有强非神经基线、独立NumPy参照、正常与-O检查、参数轨迹和逐像素预测，13张原创图。

```bash
python test_experiment.py
python -O test_experiment.py
python experiment.py --output reproduced
python make_figures.py --results reproduced/results.json --output reproduced/figures
```

普通CNN平均测试前景IoU 0.6770；3×3均值阈值0.6207；加权CNN0.6277。全背景像素准确率89.84%但前景IoU为0。背景偏移使普通CNN明显退步。三种子、合成数据与固定预算不支持普遍模型排名。

environment.yml与requirements.txt描述学习环境；outputs/environment.json记录实际版本。build_tools保留PDF和无socket Notebook构建脚本及依赖说明。独立恢复包含本单元全部源、数据、产物和课程原大纲，不包含安装缓存。

发布Notebook在新Python进程内创建真实ipykernel InProcessKernel，顺序执行全部代码，重新训练并保存PNG。当前构建未测试socket传输或浏览器Jupyter界面。verification.json记录作者检查，不宣称独立审查通过。

数值参照是有边界的教学实现：无ignore/crowd/多类检测汇总，不是完整COCO评测；主实验不是标准U-Net/FPN。严格定义、空集口径和研究限制请读lecture。

正类权重按logit的float64类型构造；实际训练入口回归测试同时核对q、dtype与独立首步更新。当前测试含10项，均可在正常和-O模式执行。
