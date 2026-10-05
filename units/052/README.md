# 052 注意力的逐项计算

先修012、019、049。从加权和、QKV与缩放，走完同一双样本实例的完整Jacobian、三条共享路径、跨位置/样本梯度累加、两次同步SGD及三轮前向。固定12配置诊断实验暴露未来泄漏，并保留合法模型没超过零基线的结果。

- lecture.md/pdf：理论、9幅原创图及三状态完整数值账本
- lab.md/pdf：独立操作、安装、练习和验收
- answers.md/pdf：逐题解释与故障定位
- experiment.ipynb：真实执行；Run All重算全部实验与图
- experiment.py、test_experiment.py：数值核心与独立验证
- data/、outputs/：固定协议、原始合成值、SHA、全部测试预测与结果
- make_figures.py、create_notebook.py、build_ledger.py：可重建源码
- build_pdf_052.py、environment.yml、sources.md：构建入口、环境与一手来源

从课程根目录：

```bash
conda env create -f units/052/environment.yml
conda activate dl052
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python units/052/test_experiment.py
python -O units/052/test_experiment.py
python units/052/experiment.py --output tmp/052-replay
python units/052/make_figures.py --output tmp/052-replay/figures
jupyter lab units/052/experiment.ipynb
```

数值核心无需字体；make_figures和完整Notebook需要Noto Sans CJK系统字体，environment.yml不会安装系统字体。默认路径/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc；缺失明确报错。Debian/Ubuntu执行sudo apt-get install fonts-noto-cjk，或从Google Noto官方发行安装，再把DL_CJK_FONT设为实际ttc/otf路径。lab提供安装、路径检查、缓存与首跑完整命令。只实测Linux；其他平台安装、浏览器UI与socket未测。

Notebook可从根目录或单元目录启动；create_notebook.py会重建未执行版本，阅读无需调用。离线真实新核复验：python shared/execute_notebook_inprocess.py units/052/experiment.ipynb。它验证顺序执行与输出，不验证浏览器交互。

PDF重建依赖见shared/build-tools/README.md。运行python units/052/build_pdf_052.py units/052/lecture.md（lab/answers同理）。图依赖Noto Sans CJK，PDF还使用Noto Serif CJK；不打包字体二进制。构建器只用于可信课程源码，不是陌生HTML/TeX的安全沙箱。

不要删除随包outputs后运行测试；新实验写入独立输出目录。CPU float64、无dropout、单轴≤512、有限输入绝对值≤10000；不支持GPU/半精度/缓存/稀疏或长上下文。有效query空支持拒绝，无效query零输出是显式约定。当前库全屏蔽行为依调用路径不同，不静默清洗NaN。

作者执行与视觉检查不等同于独立QA。所有12实验均保留，包括无提升；三种子仅描述性报告，不作普适或显著性结论。

可选PDF构建的Node锁文件为build-tools-package-lock.json。需要锁定安装时，在课程根目录将shared/build-tools/package.json与该锁文件分别复制到.build-deps/package.json及.build-deps/package-lock.json，再运行npm ci --prefix .build-deps。只安装构建依赖，不含任何用户凭据或数据。Python构建依赖由shared/build-tools/requirements.txt列出，系统Pango与字体仍需按官方平台说明安装。
