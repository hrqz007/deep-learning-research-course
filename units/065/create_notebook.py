"""Rebuild the fully executed teaching notebook; no network required."""
from pathlib import Path
import nbformat as n
R=Path(__file__).resolve().parent;cells=[]
def md(s):cells.append(n.v4.new_markdown_cell(s))
def code(s):cells.append(n.v4.new_code_cell(s))
md('# DL065 设备显存与性能测量\n\n真实按序执行，包含完整重训、可复核输出与内嵌图。所有新输出写入notebook_outputs。')
code("from pathlib import Path\nimport sys, json, copy, numpy as np, torch\nimport experiment as e\nimport matplotlib.pyplot as plt\nfrom io import BytesIO\nfrom IPython.display import Image, display\ndef show(fig):\n    b=BytesIO(); fig.savefig(b,format='png',dpi=135,bbox_inches='tight')\n    display(Image(data=b.getvalue())); plt.close(fig)\ntorch.set_num_threads(1)\nprint({'python':sys.version.split()[0],'torch':torch.__version__,'unit':Path.cwd().name,'device':'CPU'})")
md('## 1 参数与状态账本\n\n先数元素再数bytes。账本不是峰值内存。')
code("m=e.make_model();o=torch.optim.Adam(m.parameters(),foreach=False)\nprint('before',e.memory_ledger(m,o))\ne.step(m,o,*e.make_data(8))\nprint('after',e.memory_ledger(m,o))")
md('## 2 正确性测试\n\n测试相同目标、梯度与更新，不设置速度阈值。')
code("import unittest, test_experiment\nresult=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(test_experiment))\nif not result.wasSuccessful(): raise RuntimeError('unit tests failed')")
md('## 3 完整实际训练与基准\n\n重新完成320次训练更新、独立计时与profiler，写入新目录。计时会变。')
code("r=e.main(Path('notebook_outputs'))\nprint('final:',r['history'][-1])\nfor z in r['measurements']: print(z['implementation'],z['median_seconds'],z['samples_per_second'])\nprint('GPU memory:',r['gpu_memory'])")
md('## 4 实测学习曲线\n\n测试集独立生成；单次种子不是现实性能保证。')
code("fig,axes=plt.subplots(1,2,figsize=(9,3))\nh=r['history']\naxes[0].plot([z['epoch'] for z in h],[z['train_loss'] for z in h]);axes[0].set(xlabel='Epoch',ylabel='Train CE')\naxes[1].plot([z['epoch'] for z in h],[z['test_accuracy'] for z in h]);axes[1].set(xlabel='Epoch',ylabel='Test accuracy');fig.tight_layout();show(fig)")
md('## 5 稳态块而非最小一次\n\n图显示实际多个计时块；没有计入I/O或传输。')
code("fig,ax=plt.subplots(figsize=(7,3))\nax.boxplot([[v*1000 for v in z['seconds_per_step_blocks']] for z in r['measurements']],tick_labels=['Vectorized','Row loop'])\nax.set(ylabel='ms / step',title='Resident data, CPU 1 thread');show(fig)")
md('## 6 checkpoint复核\n\n只加载可信文件，恢复真实训练权重并核对终点。')
code("saved=torch.load('notebook_outputs/trained_model.pt',weights_only=True)\nm=e.make_model();m.load_state_dict(saved['state_dict']);a=np.load('notebook_outputs/data.npz')\nwith torch.no_grad(): score=float((m(torch.from_numpy(a['test_x'])).argmax(1)==torch.from_numpy(a['test_y'])).float().mean())\nprint('reload accuracy',score)\nif abs(score-r['history'][-1]['test_accuracy'])>1e-8: raise RuntimeError('checkpoint mismatch')")
md('## 7 分析器的边界\n\nself time不是父子total时间之和，事件内存不是峰值。')
code("print(r['profile'][:5])\nprint('memory scope:',r['memory_after_training']['scope'])")
nb=n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
n.write(nb,R/'experiment.ipynb');print('Notebook created')
