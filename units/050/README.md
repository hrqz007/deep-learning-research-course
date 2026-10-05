# DL050 循环网络与时间反向传播

原课程目标：状态怎样积累历史，长期依赖为何难以训练。先修：DL027、DL035、DL049。包含时间展开、完整 BPTT、Jacobian 连乘、截断、梯度裁剪及独立序列状态边界；不重设课程或缩减原目标。

## 阅读顺序

1. lecture.md / lecture.pdf：完整主线、双序列数值链与 13 幅原创图
2. lab.md / lab.pdf：可独立使用的实验指导
3. experiment.ipynb：已从全新实际 IPython 内核执行，包含实际 PNG 和真实全实验
4. answers.md / answers.pdf：逐题推导、完整 27 组结果与研究边界

## 最短复现

在本目录按 lab.md 创建隔离 Python 3.12 CPU 环境后：

```bash
python generate_data.py
python test_experiment.py
python -O test_experiment.py
python experiment.py --out .work/reproduction
```

命令从其他工作目录调用也可；数据与协议按脚本所在目录定位。生产入口运行全部 27 组，固定最终步、全部 seed 汇报，默认输出到 outputs；用 --out 避免覆盖提供的原始结果。核心数值依赖、可选界面依赖、PDF 构建依赖和字体安装分别见 requirements.txt、environment.yml、build_tools/README.md。

## 结果概览

在本次固定协议中，短延迟三种设置都成功；delay=24 的未裁剪 full 接近随机基线，而 full_clip 明显改善。截断组并非必然失败，即使其早期输入梯度为精确零；状态数值保留、共享参数更新与完整信用分配是不同问题。结论仅属于这个有限的合成任务和预算。

## 可审计内容

独立 Python math 标量前向、NumPy 矩阵 BPTT、PyTorch autograd、逐参数/逐输入有限差分、共享梯度求和、真实训练一步与下一次前向回归均可复验。正常/-O 脚本与新实际 IPython 内核已执行。浏览器 Jupyter 界面和 socket 传输未测试。所有 Tensor 创建显式使用 dtype 和 device；只支持声明的 float64 CPU 小规模输入域，明确拒绝极端或非法输入。

data/ 和 outputs/ 是科学数据，不包含个人资料。sources.md 区分外部原始来源与本包原创证据。freeze-sha256.json 固定本单元公开文件，排除自身和运行缓存；它不是质量认证。作者检查不等于独立审阅结论。
