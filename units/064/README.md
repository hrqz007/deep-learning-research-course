# DL064 表示或生成研究项目

围绕一个可证伪的VAE假设完成完整研究：相同2000次更新下，前500步线性KL预热能否改善留出负ELBO？目录先修022、040、058；选择VAE路线补060，不要求学完其他生成家族。

## 实际结果

五配对种子11、23、37、53、71。恒定β与预热各2000步，双预算恒定β4000步，共15个真实训练模型。测试512图、32次后验Monte Carlo估计；先验样本每模型1024图。

预热减恒定的平均负ELBO差+0.0458 nats/图，0/5胜，质量护栏差−0.0070。预定条件为平均改善至少0.2、4/5胜、有效率下降不超过0.02；本次不支持假设。保留负结果、全部种子、失败图和双预算对照，不根据测试回改方案。

恒定基线已使用4个潜维，打乱编码显著损害重建；完全潜变量塌缩可能不是这里主要瓶颈。五个种子只描述固定拆分下的优化变化，不能外推为所有VAE或语言模型的结论。

## 数据与权重

- data/bars.npz：原创8×8二值带噪条纹，训练1024、验证256、测试512、12个基础模板
- outputs/{constant,warmup,long_constant}_seed*.pt：15个真实权重
- outputs/*_arrays.npz：逐例NLL、KL、负ELBO、编码参数、均值解码、先验概率、二值样本与失败索引

独立像素基线测试NLL25.8346，恒定VAE平均估计负ELBO14.1203。模板有效率仅是透明代理指标；概率图、后验均值重建和真正先验二值抽样严格分开。

## 阅读与重建

- lecture.md / lecture.pdf：自包含中文教材、推导、原创建模图、真实结果与范围
- lab.md / lab.pdf：完整操作实验、手算检查、研究报告要求
- answers.md / answers.pdf：逐步答案、数值与结论边界
- experiment.ipynb：已顺序执行，含实际重训和嵌入PNG

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_experiment.py
python -O -m unittest -v test_experiment.py
python experiment.py
python make_figures.py
python create_notebook.py
python execute_notebook.py experiment.ipynb
python build_pdf.py lecture.md
python build_pdf.py lab.md
python build_pdf.py answers.md
```

推荐独立Python3.12环境，CPU即可，无外部数据或预训练模型。完整训练为分钟内至分钟量级，依硬件和负载而变；实际版本与验证见verification.json。图和PDF构建需要Noto CJK字体，PDF还需要WeasyPrint系统依赖；Poppler用于渲染检查。公式本地转SVG，无在线公式服务。

实验默认重建outputs同名文件，新实验用--output另存。Notebook重训写notebook_outputs，避免覆盖固定教材结果。执行器使用新Python进程内IPython内核，顺序保存输出，不声称测试浏览器Jupyter界面。不同平台和版本不保证逐位一致。只加载信任来源的检查点，示例使用weights_only=True。

## 文件说明

- experiment.py、test_experiment.py：注释源代码与20项测试，含普通及优化模式验证
- make_figures.py、figures/：6张原创流程、结果和诊断图
- outputs/results.json、protocol.json：全部测量和明确实验协议
- create_notebook.py、execute_notebook.py：可重建且真实执行的Notebook
- build_pdf.py、build-tools/rebuild.sh：文档构建与完整重建入口
- sources.md、source-checks.json：经核验的一手论文与官方文档
- DATA_LICENSE.md：原创数据、图文与代码授权说明
- verification.json：已实际执行的检查与适用边界

临时目录、Notebook重复训练输出、缓存和渲染质检中间图不属于固定交付。所需代码、输入、完整结果和已执行Notebook均保留。
