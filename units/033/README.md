# 033 动量学习率与调度

核心问题：惯性与随时间变化的步长怎样改变优化轨迹？直接先修：[032](../032/README.md)。本讲独立运行，不导入先修单元的代码。

## 阅读和运行顺序

1. [正文讲义](lecture.pdf)：动量与EMA、二次谷稳定性、同一2样本9参数网络的两次完整前向反向和更新、学习率端点与单位、精确恢复。
2. [单独实验指南](lab.pdf)：可执行命令、手算与故障定位。
3. [已执行Notebook](experiment.ipynb)：完整两次trace、CPU重新计算、原始结果对比及图。
4. [练习详解PDF](answers.pdf)与[可搜索答案](answers.md)。
5. [来源与查阅范围](sources.md)、[作者验证记录](verification.json)。

9幅原创图解释更新环、前向图、样本路径贡献、EMA权重和滞后、二次谷轨迹、LR曲线、累积时钟、网络结果与恢复状态。完整原始数值保留在outputs，图不会替代逐坐标核验。

## 环境

实际运行环境：Linux x86_64、Python3.12.14、NumPy2.3.5、PyTorch2.7.1+cpu、单线程float64，开启确定性算法。复用已有CPU隔离运行时，没有在用户电脑安装。无显卡、数据下载、模型下载或付费API要求。

可使用兼容环境，也可以自行创建隔离环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.3.5
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install matplotlib==3.10.8 jupyterlab ipykernel nbformat
```

Windows激活方式不同，可使用.venv\Scripts\activate。[environment.yml](environment.yml)提供Anaconda声明；没有重新验证全新conda安装或所有操作系统。选择Notebook内核时，请确认它对应安装这些包的环境。

在本目录执行：

```bash
python experiment.py --output outputs
python -m unittest -v test_experiment.py
python -O experiment.py --output optimized_outputs
jupyter lab experiment.ipynb
```

脚本根据自身路径寻找data，可从其他工作目录启动。Notebook要求在本讲目录打开并重启内核后运行全部。交付版在新的Python进程内使用IPython InProcessKernel顺序执行，未测试浏览器界面或socket传输。notebook_outputs与optimized_outputs是自行运行时的产物，不是需要另外下载的课程文件。

## 可核对的结果

- 同一手算批次：loss 0.0221 → 0.0084925618773056 → 0.00270862158258904；两次共18个参数时点有限差分检查。
- 48点小网络：四方案各72次更新、576样本呈现；固定SGD/固定动量/动量余弦/预热余弦终点loss约0.00157943/0.00047501/0.00094183/0.00091694。
- 二次函数：同起点、80更新；固定SGD与固定动量终点函数值约0.00284284和4.23578e-7。
- 检查点在epoch4末、24更新后保存；只继续原预算中剩余48次更新。完整恢复逐状态、逐日志、全部排列和最终JSON字节一致。

恢复命令适用于Bash/zsh，其他终端可去掉反斜线并连成一行：

```bash
python experiment.py \
  --resume-checkpoint outputs/checkpoint-epoch-04.json \
  --output resumed_outputs
```

主要脚本输出8文件：trace.json、quadratic.json、network.json、schedules.csv、checkpoint-epoch-04.json、final-state.json、resume.json、summary.json。它们分别保存完整教学链、轨迹、时钟、状态和摘要。检查点是课程自生成JSON，校验和检测意外改变，不构成文件真实性认证。

## 固定教材与自由探索

固定CLI在计算和输出写入前核对3输入的字节摘要，任何空白、参数或标签改动也会拒绝。绘图脚本先核对3输入和8参考结果；Notebook第一格同样校验这些文件和将展示的图片，再导入训练功能。修改数据后不要继续展示旧图。

探索时保留固定文件，调用公开的validate_batch、manual_backward、trace_steps、schedule_lr、quadratic_run或train_network，在新目录保存新实验。数组要求有限非布尔实数，绝对值≤100；X为(n,2)、y为(n,1)，1≤n≤128。train_network只接受可被微批次大小×累积次数整除的完整窗口。epoch为1至50、总更新数2至10000、momentum为[0,0.99]、峰值LR为[1e-8,0.2]，更详细边界见函数和实验指南。

固定驱动build_outputs只负责本课程固定配置；不是任意数据规模的报告生成器。forward是内部张量表达式，调用者先验证shape和范围。train_network的_omit参数只用于故意漏状态的负对照，普通训练不要使用。

所有校验、计算及序列化先完成，再建立输出目录。写入使用逐文件原子替换，不承诺多文件的断电事务；磁盘故障可能导致新旧文件混合，需重新运行并核对全部摘要。固定数据或JSON序列化失败时，不应创建或更改输出。

重建中文图需要Noto Sans CJK字体；make_figures.py使用Linux常见字体位置。其他平台可能需要调整字体路径。计算和已提供PDF/PNG不要求重新绘图。

## 边界

这是单种子、固定配置、CPU小规模机制演示，无独立验证/测试集，不比较真实任务泛化或硬件性能。恢复只覆盖完整epoch边界、单线程确定性CPU、专用采样Generator；没有验证GPU、多worker、分布式、混合精度或累积窗口中途恢复。没有Dropout与BatchNorm，且初始参数写死，所以训练不使用全局PyTorch/NumPy随机源；引入新随机算子后必须重新识别需保存的状态。
