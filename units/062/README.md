# DL062 离散扩散与去噪学习

逐步加噪为什么可以反过来生成数据？本单元从马尔可夫分解、条件高斯和变分界开始，完整连接前向闭式、后验、KL与噪声预测，训练一个真正从标准正态生成二维点的DDPM。

这里时间是离散整数，状态是连续二维实数；不是离散词元扩散。目录先修：017、018、019、030、032、035；本讲补齐所需概率衔接，不要求先做完整VAE实验。

## 阅读与运行

- lecture.pdf / lecture.md：完整中文教材与数学推导、图、实测结果。
- lab.pdf / lab.md：从环境、公式测试到真实训练和研究设计。
- answers.pdf / answers.md：逐步推导、数值答案与结论边界。
- experiment.ipynb：已顺序执行，包含三种子完整训练。

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_experiment.py
python -O -m unittest -v test_experiment.py
python experiment.py
python make_figures.py
python create_notebook.py
python execute_notebook.py experiment.ipynb
```

推荐独立Python3.11/3.12环境、CPU运行。实际版本记录在verification.json；无需外部数据、账户、预训练模型或GPU。完整运行通常为分钟量级，取决于硬件。同名输出会被重建覆盖，扩展实验请用--output另存。

## 实测结果

T=200，β从0.0001到0.05，barαT=0.006121965。三种子11、23、37各7000次更新，批量256。最终均8/8覆盖，有效比例0.9846、0.9834、0.9927，验证噪声MSE0.2909、0.2987、0.2868；零预测基线0.9801。

网络只用xt和t预测ε。训练用原始点与主动抽取噪声，采样从标准正态开始且不读取真实点。样本、完整配置、阶段性MSE、按时间区间MSE和权重均保存。

训练日志是无权重、按坐标平均的噪声MSE，不是精确似然或完整ELBO。默认末步返回均值；terminal_noise=True按本课正方差连续端点密度抽样。两种结果另存并分别报告。前向终点只是近似标准正态，barαT不是0。

## 文件说明

- experiment.py：日程、闭式、后验/反向均值、时间嵌入、训练和采样。
- test_experiment.py：18项数学、随机矩、索引、形状、反向循环与梯度测试。
- make_figures.py、figures/：6张原创流程及实测图，含前向边缘图。
- data/mixture.npz：原创训练、独立参考数据、评价中心。
- outputs/results.json：实际运行全量摘要与轨迹。
- outputs/schedule.npz：与数学t=0..T一致的完整日程。
- outputs/ddpm_seed*.pt：真实训练的ε网络参数和配置。
- outputs/ddpm_seed*_samples.npz：默认样本、正方差末步样本、反向中间状态。
- create_notebook.py、execute_notebook.py：可重建的Notebook与无socket顺序执行器。
- build_pdf.py、build-tools/rebuild.sh：本地文档构建与重建入口。
- sources.md、source-checks.json：核验的一手论文与官方文档。
- verification.json：实际验证证据与已知边界。
- DATA_LICENSE.md：合成数据、原创图和代码授权。

Notebook重跑写入notebook_outputs，避免覆盖教材outputs；重复缓存、tmp与`__pycache__`不在固定交付清单内。只加载你信任来源的pt文件，可使用PyTorch受限weights_only加载方式。

## 重建PDF

```bash
python build_pdf.py lecture.md
python build_pdf.py lab.md
python build_pdf.py answers.md
```

需要Noto CJK中文字体与WeasyPrint系统依赖；Poppler用于独立渲染检查。公式本地转SVG，不依赖在线服务。Notebook验证采用新IPython进程的in-process内核，不声称测试浏览器Jupyter界面。

本单元没有实现图像U-Net、条件生成、引导或加速采样。二维结果不等于图像基准成绩；与GAN比较时训练及采样预算不同，不能视为公平排行榜。
