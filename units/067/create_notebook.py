"""Rebuild the fully executed teaching notebook; no network required."""
from pathlib import Path
import nbformat as n
R=Path(__file__).resolve().parent;cells=[]
def md(s):cells.append(n.v4.new_markdown_cell(s))
def code(s):cells.append(n.v4.new_code_cell(s))
md('# DL067 分布式训练的基本语义\n\n真实按序执行，包含完整重训、可复核输出与内嵌图。所有新输出写入notebook_outputs。')
code("from pathlib import Path\nimport sys, json, copy, numpy as np, torch\nimport experiment as e\nimport matplotlib.pyplot as plt\nfrom io import BytesIO\nfrom IPython.display import Image, display\ndef show(fig):\n    b=BytesIO(); fig.savefig(b,format='png',dpi=135,bbox_inches='tight')\n    display(Image(data=b.getvalue())); plt.close(fig)\ntorch.set_num_threads(1)\nprint({'python':sys.version.split()[0],'torch':torch.__version__,'unit':Path.cwd().name,'device':'CPU'})")
md('## 1 全局目标与不等rank\n\n这里的rank是模型副本，没有网络。')
code("x,y=e.data();m=e.model();parts=[torch.arange(20),torch.arange(20,50),torch.arange(50,128)]\nrows=[e.gradient(copy.deepcopy(m),x[p],y[p]) for p in parts]\nfull=e.flatten(e.gradient(copy.deepcopy(m),x,y)[0])\nfor mode in ['weighted','rank_mean','double_divide']:\n    g=e.flatten(e.average_gradients([z[0] for z in rows],[z[1] for z in rows],mode))\n    print(mode,float((g-full).abs().max()))")
md('## 2 实际采样器索引\n\n调用DistributedSampler不等于启动了DDP。')
code("for drop in [False,True]:\n    parts=e.sampler_indices(drop_last=drop);flat=sum(parts,[])\n    print('drop_last',drop,parts,'rows',len(flat),'unique',len(set(flat)))\nprint('epoch0',e.sampler_indices(shuffle=True,epoch=0));print('epoch1',e.sampler_indices(shuffle=True,epoch=1))")
md('## 3 数学语义与边界测试\n\n含掩码、局部零计数、无效全局计数与采样覆盖。')
code("import unittest, test_experiment\nresult=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(test_experiment))\nif not result.wasSuccessful(): raise RuntimeError('unit tests failed')")
md('## 4 真正训练但仅模拟通信\n\n执行三种子四模式，每个80次更新。')
code("r=e.main(Path('notebook_outputs'))\nprint(r['execution_kind'])\nfor z in r['runs']: print(z['seed'],z['history'][-1])")
md('## 5 训练曲线与权重误差\n\n错误rank平均改变权重，重复rank改变覆盖。')
code("fig,ax=plt.subplots(figsize=(8,3))\nh=r['runs'][0]['history']\nfor mode in ['full','weighted','rank_mean','duplicated_rank']:ax.plot([z['step'] for z in h],[z[mode]['test_loss'] for z in h],label=mode)\nax.set(xlabel='Update',ylabel='Test CE',title='Numerical simulation; no network');ax.legend();show(fig)\nprint('parameter errors',[z['history'][-1]['full_weighted_parameter_max_error'] for z in r['runs']])")
md('## 6 通信公式不是计时\n\n100MiB/12.5GB/s/5微秒假设下的理想环模型。')
code("fig,ax=plt.subplots(figsize=(7,3))\na=r['communication_model'];ax.plot([z['world'] for z in a],[z['seconds']*1000 for z in a],'o-')\nax.set(xlabel='World size',ylabel='Modelled ring time (ms)',title='Formula only, not measured communication');show(fig)\nprint(r['real_distributed_run'])")
md('## 7 全部终点权重复核\n\n统一FP64结构，在保存的测试集计算。')
code('a=np.load(\'notebook_outputs/data.npz\')\nfor row in r[\'runs\']:\n    for mode in [\'full\',\'weighted\',\'rank_mean\',\'duplicated_rank\']:\n        z=torch.load(f"notebook_outputs/{mode}_seed{row[\'seed\']}.pt",weights_only=True);m=e.model(z[\'seed\']);m.load_state_dict(z[\'state_dict\'])\n        with torch.no_grad():v=float(torch.nn.functional.cross_entropy(m(torch.from_numpy(a[\'test_x\'])),torch.from_numpy(a[\'test_y\'])))\n        if abs(v-row[\'history\'][-1][mode][\'test_loss\'])>1e-12:raise RuntimeError(\'checkpoint mismatch\')\nprint(\'12 checkpoints reproduced\')')
nb=n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
n.write(nb,R/'experiment.ipynb');print('Notebook created')
