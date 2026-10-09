"""生成可从空内核完整执行的教学Notebook；输出需随后真实执行。"""
from pathlib import Path  # 输出定位到本单元。
import nbformat  # 官方Notebook数据格式。
ROOT=Path(__file__).resolve().parent
nb=nbformat.v4.new_notebook()  # 不带缓存输出的新文档。
cells=[]
def md(text): cells.append(nbformat.v4.new_markdown_cell(text))  # 教学说明。
def code(text): cells.append(nbformat.v4.new_code_cell(text))  # 可执行代码。
md('# DL074 统计学习与泛化界\n\n发行074映射原大纲T2。本Notebook从空内核实际重做完整实验。先预测结果，再逐单元执行；所有数据原创、CPU离线。图由固定outputs绘制，下面先核对重跑数组再展示。')
code("from pathlib import Path  # 当前目录应为本单元。\nimport json, numpy as np  # 结果读取与数值核验。\nfrom IPython.display import display, Image  # 内嵌PNG到Notebook。\nprint('working directory:', Path.cwd().name)  # 只显示目录名，避免机器路径。")
md('## 1 先做手算可核验的机制\n数组形状与定义见data/README.md。不要把程序输出当作理解推导的替代。')
code("from generate_data import population, sample  # 固定总体和独立抽样。\nfrom experiment import radius, true_risks, empirical_risks  # 分开三类数字。\npop = population()  # 不用本次样本决定候选集合。\nidx, y = sample(100, 7400)  # 100条独立样本。\ntruth = true_risks(pop['predictions'], pop['prob'])  # 已知总体精确求和。\nemp = empirical_risks(pop['predictions'], idx, y)  # 经验风险。\nprint('radius:', radius(100, 23), 'max gap:', np.max(abs(emp-truth)))  # 真正计算。\nprint('candidate and distinct counts:', len(emp), len(np.unique(pop['predictions'], axis=0)))")
code("best = int(np.argmin(emp))  # 并列选最小索引。\nprint('chosen, empirical, population:', best, emp[best], truth[best])  # 不混淆两个风险。\nnp.testing.assert_allclose(truth.min(), .15, atol=1e-14)  # 标签噪声的可核验基准。\nprint('n for radius <= .05:', int(np.ceil(np.log(2*23/.05)/(2*.05**2))))  # 足够样本量。")
md('## 2 完整实验重跑\n调用run确实重新计算全部预定配置；输出另存notebook_outputs。运行时间只反映当前CPU，不构成跨机器性能比较。')
code("from experiment import run  # 与命令行相同完整入口。\nresult = run('notebook_outputs')  # 真实重新抽样或训练，不复用固定输出。\nprint('unit:', result['unit'], 'runtime:', result['runtime'])  # 保存真实版本与时间。")
code("for row in result['rows']:  # 每个样本量分别报告。\n    print(row['n'], 'radius:', row['epsilon'], 'violations:', row['violation_count'])\n    print('ERM train/population:', row['mean_train'], row['mean_true'])  # 选择偏差可见。")
md('## 3 逐数组核对固定发行结果\n同环境重跑应相同；跨BLAS或平台可能有舍入差，使用小容差。NaN仅在有明确定义的未定义量处按相同位置匹配。')
code("reference = np.load('outputs/trials.npz', allow_pickle=False)  # 固定发行证据。\nreplayed = np.load('notebook_outputs/trials.npz', allow_pickle=False)  # 刚重跑结果。\nif set(reference.files) != set(replayed.files):  # 不依赖可被-O删掉的assert。\n    raise ValueError('array keys differ')\nfor name in reference.files:  # 每个数组都检查，不只检查摘要。\n    np.testing.assert_allclose(replayed[name], reference[name], rtol=1e-10, atol=1e-11, equal_nan=True)\nprint('checked arrays:', len(reference.files))  # 实际核验数量。")
md('## 4 读图并写出限制\n图是make_figures.py由固定outputs产生；上一单元已核对刚重跑的全部数组。改变实验后应重新画图，不能继续把旧图当新结果。')
code("display(Image(filename='figures/03_bound.png'))  # 内嵌真实结果图。")
code("display(Image(filename='figures/01_uniform.png'))  # 机制图不是额外实验结果。")
md('## 5 自检\n回到lab完成故障注入与扩展设计。明确已执行和仅提出的实验，解释至少一条不能外推的条件。答案在answers.md，不使用测试结果挑选最好配置。')
nb.cells=cells
nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
nbformat.write(nb,ROOT/'experiment.ipynb')  # 随后execute_notebook.py真实执行。
