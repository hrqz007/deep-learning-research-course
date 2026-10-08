# DL072 独立研究项目与审查

一个完整可独立运行的中文研究教学项目：打乱训练图像角标，能否减少模型对捷径的依赖？从问题、相关工作和训练目标，走到五方法、三种子、消融、配对统计、拒答、预算与限制。

真实训练15个PyTorch模型。角标随机化把预定反转域平均错误率从68.28%降到18.22%，但同分布错误率增加12.44个百分点，核心加噪域增加17.61个百分点；简单遮挡给出相近均值。报告保留这些代价和替代解释，不宣称方法创新、发表价值或任意域安全。

## 学习与研究材料

- lecture.md / lecture.pdf：9页中文教材、目标推导、6张原创彩色图和完整实测讨论
- lab.md / lab.pdf：独立重跑、手算、代码故障植入与研究报告任务
- answers.md / answers.pdf：数值答案、消融解释与审查反馈
- research_plan.json：固定问题、数据、方法、种子、预算和指标的协议，非外部预注册
- project_report.md：完整短研究报告范例，含相关工作、结果、预算和限制
- experiment.ipynb：已从零顺序执行15个模型训练并内嵌结果图

## 独立运行

在本单元目录运行，不需要其他课程单元、外部数据、账号或GPU。推荐独立Python3.12环境；也可使用environment.yml创建Anaconda环境。已验证版本见verification.json。

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_experiment.py
python -O -m unittest -v test_experiment.py
python experiment.py --output my_project
```

前两条测试命令各17项。默认实验5方法×3种子×160次全批更新，共2400次更新；CPU单线程，通常数秒完成。首次安装依赖可能联网，实际实验完全离线。

自己的运行建议另存my_project；默认输出目录outputs会覆盖同名文件。全部五方法固定报告，没有超参数搜索、早停或最佳种子选择。

## 原始证据

outputs/data.npz保存完整800训练、240验证、600测试样本与四域配对输入；predictions.npz包含逐例概率。results.json保存损失轨迹、逐次与聚合指标、配对差值、选择性结果、子群、预算和环境。15份.pt是真实训练权重，全部保存后重载核对预测最大差为0。只加载可信文件并使用weights_only=True；checkpoint仅供推理复核，没有完整续训状态。

三种子统计只覆盖固定数据上的初始化／增强随机性。四个测试域共享原始样本，不能视为四份独立现实数据。计时是本机观察，训练数组字节数不是峰值内存。

## Notebook和图

```bash
python create_notebook.py
python execute_notebook.py experiment.ipynb
python make_figures.py
```

Notebook另存notebook_outputs，实际完整重训，不覆盖固定outputs。执行器使用新Python进程内的IPython内核，未验证浏览器Jupyter或外进程socket传输。make_figures.py从固定outputs重建六张图。中文字体默认Noto CJK，也可用COURSE_CJK_FONT指定字体路径。

## PDF重建

```bash
python build_pdf.py lecture.md
python build_pdf.py lab.md
python build_pdf.py answers.md
```

build-tools/rebuild.sh串联完整流程。PDF构建还需要Pango等系统组件及Noto CJK字体，阅读现有PDF无需构建工具。数学公式本地生成。

来源见sources.md与source-checks.json，原创内容和数据授权见DATA_LICENSE.md。不要把临时缓存、渲染页或重复Notebook输出作为课程文件提交。
