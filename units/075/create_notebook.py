"""生成可从空内核完整执行的教学Notebook；输出需随后真实执行。"""
from pathlib import Path  # 输出定位到本单元。
import nbformat  # 官方Notebook数据格式。
ROOT=Path(__file__).resolve().parent
nb=nbformat.v4.new_notebook()  # 不带缓存输出的新文档。
cells=[]
def md(text): cells.append(nbformat.v4.new_markdown_cell(text))  # 教学说明。
def code(text): cells.append(nbformat.v4.new_code_cell(text))  # 可执行代码。
md('# DL075 宽网络核视角与特征学习\n\n发行075映射原大纲T3。本Notebook从空内核实际重做完整实验。先预测结果，再逐单元执行；所有数据原创、CPU离线。图由固定outputs绘制，下面先核对重跑数组再展示。')
code("from pathlib import Path  # 当前目录应为本单元。\nimport json, numpy as np  # 结果读取与数值核验。\nfrom IPython.display import display, Image  # 内嵌PNG到Notebook。\nprint('working directory:', Path.cwd().name)  # 只显示目录名，避免机器路径。")
md('## 1 先做手算可核验的机制\n数组形状与定义见data/README.md。不要把程序输出当作理解推导的替代。')
code("from generate_data import generate  # 原创固定训练点。\nfrom experiment import initialize, forward_jacobian  # 手写网络与Jacobian。\nd = generate()  # 数据无需下载。\ntheta0 = initialize(8, 11)  # 24个参数。\nf0, j0, h0 = forward_jacobian(theta0, d['x'])  # 输出、偏导、表示。\nk0 = j0 @ j0.T  # 对称半正定Gram。\nprint('f, J, H, K shapes:', f0.shape, j0.shape, h0.shape, k0.shape)  # 检查每个维度。")
code("eta = .3 / np.linalg.eigvalsh(k0/len(d['y']))[-1]  # 固定核安全谱尺度。\ndelta = -eta * j0.T @ (f0-d['y']) / len(d['y'])  # 首次更新。\nf_real, _, _ = forward_jacobian(theta0+delta, d['x'])  # 真网络含高阶项。\nf_lin = f0 + j0 @ delta  # 局部模型的一步精确值。\nnp.testing.assert_allclose(f_lin, f0-eta*k0@(f0-d['y'])/len(d['y']), atol=1e-12)\nprint('one-step prediction RMSE:', np.sqrt(np.mean((f_real-f_lin)**2)))  # 不假设为0。")
md('## 2 完整实验重跑\n调用run确实重新计算全部预定配置；输出另存notebook_outputs。运行时间只反映当前CPU，不构成跨机器性能比较。')
code("from experiment import run  # 与命令行相同完整入口。\nresult = run('notebook_outputs')  # 真实重新抽样或训练，不复用固定输出。\nprint('unit:', result['unit'], 'runtime:', result['runtime'])  # 保存真实版本与时间。")
code("for amplitude in [.2, 1., 4.]:  # 全部幅度，不挑最佳。\n    for width in [8, 32, 128]:  # 全部宽度。\n        rows = [r for r in result['rows'] if r['width']==width and r['amplitude']==amplitude]\n        print(amplitude, width, np.mean([r['relative_prediction_rmse'] for r in rows]))  # 三种子均值。")
md('## 3 逐数组核对固定发行结果\n同环境重跑应相同；跨BLAS或平台可能有舍入差，使用小容差。NaN仅在有明确定义的未定义量处按相同位置匹配。')
code("reference = np.load('outputs/trajectories.npz', allow_pickle=False)  # 固定发行证据。\nreplayed = np.load('notebook_outputs/trajectories.npz', allow_pickle=False)  # 刚重跑结果。\nif set(reference.files) != set(replayed.files):  # 不依赖可被-O删掉的assert。\n    raise ValueError('array keys differ')\nfor name in reference.files:  # 每个数组都检查，不只检查摘要。\n    np.testing.assert_allclose(replayed[name], reference[name], rtol=1e-10, atol=1e-11, equal_nan=True)\nprint('checked arrays:', len(reference.files))  # 实际核验数量。")
md('## 4 读图并写出限制\n图是make_figures.py由固定outputs产生；上一单元已核对刚重跑的全部数组。改变实验后应重新画图，不能继续把旧图当新结果。')
code("display(Image(filename='figures/04_scan.png'))  # 内嵌真实结果图。")
code("display(Image(filename='figures/01_comparison.png'))  # 机制图不是额外实验结果。")
md('## 5 自检\n回到lab完成故障注入与扩展设计。明确已执行和仅提出的实验，解释至少一条不能外推的条件。答案在answers.md，不使用测试结果挑选最好配置。')
nb.cells=cells
nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
nbformat.write(nb,ROOT/'experiment.ipynb')  # 随后execute_notebook.py真实执行。
