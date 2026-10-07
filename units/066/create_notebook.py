"""Rebuild the fully executed teaching notebook; no network required."""
from pathlib import Path
import nbformat as n
R=Path(__file__).resolve().parent;cells=[]
def md(s):cells.append(n.v4.new_markdown_cell(s))
def code(s):cells.append(n.v4.new_code_cell(s))
md('# DL066 混合精度与激活重计算\n\n真实按序执行，包含完整重训、可复核输出与内嵌图。所有新输出写入notebook_outputs。')
code("from pathlib import Path\nimport sys, json, copy, numpy as np, torch\nimport experiment as e\nimport matplotlib.pyplot as plt\nfrom io import BytesIO\nfrom IPython.display import Image, display\ndef show(fig):\n    b=BytesIO(); fig.savefig(b,format='png',dpi=135,bbox_inches='tight')\n    display(Image(data=b.getvalue())); plt.close(fig)\ntorch.set_num_threads(1)\nprint({'python':sys.version.split()[0],'torch':torch.__version__,'unit':Path.cwd().name,'device':'CPU'})")
md('## 1 三种数值格式\n\n亲自看下溢、溢出与1附近的刻度。')
code('v=torch.tensor([1e-8,1.,1.+2**-10,70000.])\nfor t in [torch.float32,torch.float16,torch.bfloat16]:print(t,v.to(t).float(),torch.finfo(t).eps)')
md('## 2 不等长微批\n\n31、47、50微批须按计数加权。')
code("x,y=e.data();m=e.model();_,a=e.gradients(m,x,y)\nfor wrong in [False,True]:\n    _,b=e.gradients(copy.deepcopy(m),x,y,sizes=[31,47,50],wrong_equal=wrong)\n    print('wrong_equal',wrong,'relative error',e.relative_error(a,b))")
md('## 3 正确与错误分支测试\n\n包括Dropout随机状态和BatchNorm反例。')
code("import unittest, test_experiment\nresult=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(test_experiment))\nif not result.wasSuccessful(): raise RuntimeError('unit tests failed')")
md('## 4 全部三种子四模式重训\n\n真实执行12次160步训练。CPU BF16实际运行；CUDA FP16未运行。')
code("r=e.main(Path('notebook_outputs'))\nfor z in r['runs']:print(z['seed'],z['mode'],z['final_test_mse'])\nprint(r['cuda_fp16_gradscaler'])")
md('## 5 完整学习曲线\n\n所有终点评价统一FP32推理。')
code("fig,ax=plt.subplots(figsize=(8,3))\nfor z in r['runs']:\n    if z['seed']==11:ax.plot([h['step'] for h in z['history']],[h['test_mse_fp32_inference'] for h in z['history']],label=z['mode'])\nax.set(xlabel='Update',ylabel='Test MSE in FP32',title='Seed 11; curves may overlap');ax.legend();show(fig)")
md('## 6 保存载荷与时间\n\nhooks记录与时间测量在独立运行中。两者都不是GPU峰值。')
code("fig,axes=plt.subplots(1,2,figsize=(9,3))\nkeys=['full','checkpoint']\naxes[0].bar(keys,[r['saved_tensors'][k]['saved_nonparameter_unique_storage_bytes']/1024 for k in keys]);axes[0].set(ylabel='Saved nonparameter storage (KiB)')\naxes[1].bar(keys,[r['timing'][k]['median_seconds']*1000 for k in keys]);axes[1].set(ylabel='CPU forward+backward ms');fig.tight_layout();show(fig)\nprint('dropout',r['dropout']);print('BatchNorm',r['batchnorm_accumulation_relative_error'])")
md('## 7 复核全部权重\n\n终点权重可复核，但没有完整续训状态。')
code('a=np.load(\'notebook_outputs/data.npz\')\nfor row in r[\'runs\']:\n    z=torch.load(f"notebook_outputs/{row[\'mode\']}_seed{row[\'seed\']}.pt",weights_only=True)\n    m=e.model(z[\'seed\']);m.load_state_dict(z[\'state_dict\'])\n    with torch.no_grad():value=float(torch.nn.functional.mse_loss(m(torch.from_numpy(a[\'test_x\'])),torch.from_numpy(a[\'test_y\'])))\n    if abs(value-row[\'final_test_mse\'])>1e-7:raise RuntimeError(\'checkpoint mismatch\')\nprint(\'12 checkpoints reproduced\')')
nb=n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
n.write(nb,R/'experiment.ipynb');print('Notebook created')
