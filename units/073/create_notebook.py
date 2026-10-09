"""生成可从空内核完整执行的教学Notebook；输出需随后真实执行。"""
from pathlib import Path  # 输出定位到本单元。
import nbformat  # 官方Notebook数据格式。
ROOT=Path(__file__).resolve().parent
nb=nbformat.v4.new_notebook()  # 不带缓存输出的新文档。
cells=[]
def md(text): cells.append(nbformat.v4.new_markdown_cell(text))  # 教学说明。
def code(text): cells.append(nbformat.v4.new_code_cell(text))  # 可执行代码。
md('# DL073 优化收敛与隐式偏置\n\n发行073映射原大纲T1。本Notebook从空内核实际重做完整实验。先预测结果，再逐单元执行；所有数据原创、CPU离线。图由固定outputs绘制，下面先核对重跑数组再展示。')
code("from pathlib import Path  # 当前目录应为本单元。\nimport json, numpy as np  # 结果读取与数值核验。\nfrom IPython.display import display, Image  # 内嵌PNG到Notebook。\nprint('working directory:', Path.cwd().name)  # 只显示目录名，避免机器路径。")
md('## 1 先做手算可核验的机制\n数组形状与定义见data/README.md。不要把程序输出当作理解推导的替代。')
code("from generate_data import generate  # 原创可分数据与二次曲率。\nfrom experiment import logistic, descent_quadratic  # 同源码的可核验数学函数。\nd = generate()  # 无下载，确定性生成。\nloss, grad = logistic(np.zeros(2), d['x'], d['y'])  # 零点应给log2和[-.55,-.6]。\nprint('zero loss, gradient:', loss, grad)  # 不手工写假输出。\nnp.testing.assert_allclose(grad, [-.55, -.6], atol=1e-14)  # 手算独立锚点。")
code("points, values = descent_quadratic(d['q'], d['w0'], .25, 50)  # 真正更新50次。\nbound = 40 / np.arange(1, 51)  # t=0无界，排除。\nprint('maximum observed minus bound:', np.max(values[1:]-bound))  # 应非正。\nnp.testing.assert_array_less(values[1:], bound + 1e-12)  # 容忍浮点误差。")
md('## 2 完整实验重跑\n调用run确实重新计算全部预定配置；输出另存notebook_outputs。运行时间只反映当前CPU，不构成跨机器性能比较。')
code("from experiment import run  # 与命令行相同完整入口。\nresult = run('notebook_outputs')  # 真实重新抽样或训练，不复用固定输出。\nprint('unit:', result['unit'], 'runtime:', result['runtime'])  # 保存真实版本与时间。")
code("for row in result['logistic']['runs']:  # 保留全部初始化，不选最佳。\n    print(row['initial'], 'angle:', row['angle_at_10'], '->', row['final_angle_deg'])\nprint('L, eta:', result['logistic']['L'], result['logistic']['eta'])  # 实际曲率尺度。")
md('## 3 逐数组核对固定发行结果\n同环境重跑应相同；跨BLAS或平台可能有舍入差，使用小容差。NaN仅在有明确定义的未定义量处按相同位置匹配。')
code("reference = np.load('outputs/trajectories.npz', allow_pickle=False)  # 固定发行证据。\nreplayed = np.load('notebook_outputs/trajectories.npz', allow_pickle=False)  # 刚重跑结果。\nif set(reference.files) != set(replayed.files):  # 不依赖可被-O删掉的assert。\n    raise ValueError('array keys differ')\nfor name in reference.files:  # 每个数组都检查，不只检查摘要。\n    np.testing.assert_allclose(replayed[name], reference[name], rtol=1e-10, atol=1e-11, equal_nan=True)\nprint('checked arrays:', len(reference.files))  # 实际核验数量。")
md('## 4 读图并写出限制\n图是make_figures.py由固定outputs产生；上一单元已核对刚重跑的全部数组。改变实验后应重新画图，不能继续把旧图当新结果。')
code("display(Image(filename='figures/04_directions.png'))  # 内嵌真实结果图。")
code("display(Image(filename='figures/01_proof.png'))  # 机制图不是额外实验结果。")
md('## 5 自检\n回到lab完成故障注入与扩展设计。明确已执行和仅提出的实验，解释至少一条不能外推的条件。答案在answers.md，不使用测试结果挑选最好配置。')
nb.cells=cells
nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
nbformat.write(nb,ROOT/'experiment.ipynb')  # 随后execute_notebook.py真实执行。
