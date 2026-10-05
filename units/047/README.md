# 047 视觉Transformer与混合结构

正式先修043与053。编号不代表学习顺序；注意力基础尚未学过时，先按049→052→053再回到本讲。保留完整102单元原先修路线。

## 内容

- [讲义PDF](lecture.pdf)与[Markdown](lecture.md)：patch/卷积等价、位置、完整小ViT/混合结构、九参数逐样本完整前后向与两轮更新、真实预算与资源边界。
- [独立实验PDF](lab.pdf)与[Markdown](lab.md)：运行、窗口核对、完整模型参照、18次比较与提交步骤。
- [练习详解PDF](answers.pdf)与[Markdown](answers.md)：12题完整解答。
- [真实执行Notebook](experiment.ipynb)、[主脚本](experiment.py)、[独立测试](test_experiment.py)、[原创数据生成器](generate_data.py)、[11幅图的生成器](make_figures.py)。
- [数据说明](data/README.md)、[来源](sources.md)、[验证范围](verification.json)。

## 运行

使用environment.yml建立独立环境。实际Linux CPU、Python3.12.14、NumPy2.3.5、PyTorch2.7.1+cpu、Matplotlib3.10.8。CPU版可在激活环境后按官方索引安装：

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python test_experiment.py
python -O test_experiment.py
python experiment.py --output rerun
python -O experiment.py --output rerun-optimized
python make_figures.py --output rerun-figures
```

数据相对脚本定位；显式相对输出路径相对当前工作目录。全离线、原创合成图、无预训练下载/API/外部图片。Notebook从本单元或仓库根启动，首格核对20个源码/数据/结果/图摘要，真正重跑18次训练并核对全部3618参数状态，保存7代码格和4实际PNG。执行使用新Python进程内真实IPython InProcessKernel，不声称验证外进程socket或浏览器UI。

绘图需要Noto Sans CJK，可通过系统官方fonts-noto-cjk或[官方Noto CJK仓库](https://github.com/notofonts/noto-cjk)取得，DL_CJK_FONT可指定字体文件；未打包字体二进制。PDF可直接阅读，重建工具和依赖在shared/build-tools。没有验证GPU或跨系统Anaconda全新安装。

## 真实比较与局限

32/128嵌套训练图×CNN/ViT/混合×3初始化种子，共18次、3600更新、288000图呈现；682/802/938参数。匹配各规模内的更新与图呈现，不匹配FLOPs、墙钟、参数或各架构最优超参。跨规模时全批次暴露量同时变4倍，不能叫纯数据效率曲线。

完整训练轨迹（3618个参数状态/NLL/准确率）、18最终state_dict、全部每图测试logit/NLL保留。首次神经比较之后增加了明确标为事后诊断的无参数邻接方向规则，未改变/重选神经结果；它在本数据100%，分数不是概率，不报NLL。

小ViT在128图上约52.73%，比32图的63.93%差；CNN准确率较高但NLL接近log2；所有失败及种子分歧保留。该8×8局部方向问题不支持真实视觉、预训练或大规模架构普遍排名。

resource_audit是真实CPU autograd保存Tensor的逻辑字节/去重storage；包含输入/参数引用，不等于分配器峰值、RSS或纯激活。CUDA不可用，peak_vram_bytes=null；显存未测，解析注意力容量不冒充实测。GPU补测方法见讲义。

## 文件与支持域

outputs/results.json保存手算全部中间量、协议/指标/预测、资源与事后基线。training_traces.json.gz完整保存每步参数与训练读数，final_states.json.gz保存可加载命名权重；results记录压缩和解压原文摘要。每文件原子替换，不声称整个输出目录事务。

patch API要求有限实数非空NCHW、正整数P、空间严格整除、最多100万元素；不会自动补边/裁剪。正式模型固定8×8/16位置和CPU float64，不实现一般图像解码、任意分辨率位置插值或所有注意力后端。版本变化可能改变数值末位及资源保存策略，直接依赖版本不构成完整跨平台锁文件。

作者检查、独立QA与远端公开分开，实际发布状态以课程发布记录为准。
