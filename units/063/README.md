# DL063 生成模型评价与风险

怎样分别检验质量、覆盖、多样性、条件一致与记忆？本课在原创二维八团上真实重训VAE与GAN各三种子，构建有限样本、特征盲点和近邻复制控制。目录先修022、019、030；只补实际比较的VAE与GAN接口。

## 实际结果

每模型4000次主更新、批量128、4096点评价。VAE有效比例0.7107、0.7065、0.7046；GAN为0.7402、0.7654、0.7673，均8/8覆盖。坐标高斯距离却偏好VAE，加入角度特征后排序改变。它是教学高斯特征距离，不是Inception FID。

预算匹配真实批次和主模型更新数，未匹配FLOPs、总优化器调用或调参预算。复制控制质量覆盖均高，但精确训练近邻匹配率1；六个模型本次匹配率均0，不构成隐私保证。模型均无条件，条件一致性仅给出协议设计，没有伪造实测。

## 数据与权重

- data/mixture.npz：训练2048、大小匹配近邻留出2048、特征参考4096、独立真实控制4096
- outputs/vae_seed*.pt、gan_seed*.pt：六个实际训练检查点，GAN含判别器状态
- outputs/*_samples.npz：完整生成点、VAE解码均值与训练/留出近邻距离
- outputs/*_control.npz：真实抽样、复制、单模式与旋转模式四种明确控制

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
