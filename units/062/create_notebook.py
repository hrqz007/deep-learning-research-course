"""Build executable DDPM teaching notebook, including three full training runs."""
from pathlib import Path
import nbformat as n
R=Path(__file__).resolve().parent;cells=[]
def md(x):cells.append(n.v4.new_markdown_cell(x))
def code(x):cells.append(n.v4.new_code_cell(x))
md('# DL062 离散扩散与去噪学习\n\n时间离散、状态连续。此Notebook在新内核顺序执行，真实训练三种子DDPM。训练与采样严格区分；输出写入notebook_outputs，不覆盖教材固定结果。')
code("from pathlib import Path\nimport sys, json, numpy as np, torch\nimport experiment as e\nfrom io import BytesIO\nfrom IPython.display import Image, display\ndef show(fig):\n    buffer = BytesIO()\n    fig.savefig(buffer, format='png', dpi=130, bbox_inches='tight')\n    display(Image(data=buffer.getvalue()))\ntorch.set_num_threads(1)\nprint({'python':sys.version.split()[0],'torch':torch.__version__,'unit':Path.cwd().name})")
md('## 1 检查索引与日程\n数组长度T+1，索引0表示干净端点；训练与反向预测只用1到T。β是方差，采样必须乘平方根。')
code("s=e.schedule()\nprint('T:',s['T'],'length:',len(s['beta']))\nprint('beta_0, alpha_bar_0:',s['beta'][0].item(),s['abar'][0].item())\nprint('alpha_bar_T:',s['abar'][-1].item(),'posterior_var_1:',s['posterior_var'][1].item())")
md('## 2 手算两步前向与后验\nβ1=0.1、β2=0.2。x0=2时，x2均值为2√0.72，方差0.28；已知x2=1时，上一步后验方差约0.07143。')
code("toy=e.schedule(2,.1,.2)\nx0=torch.full((1,2),2.)\nprint('forward sample epsilon=-1:',e.q_sample(x0,torch.tensor([2]),-torch.ones_like(x0),toy))\nprint('posterior mean for x2=1:',e.posterior_mean(torch.ones_like(x0),x0,torch.tensor([2]),toy))\nprint('posterior variance:',toy['posterior_var'][2].item())")
md('## 3 公式与边界测试\n比较闭式/链式矩、后验两种表达、KL权重、t=0恒等、t=1分支与完整反向时间序列。')
code("import unittest,test_experiment\nresult=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(test_experiment))\nif not result.wasSuccessful(): raise RuntimeError('unit tests failed')")
md('## 4 数据与前向边缘\n每幅图是给定时间的边缘抽样，不声称四幅图构成同一条逐步前向链。')
code("import matplotlib.pyplot as plt\nx,_=e.make_data();tx=torch.from_numpy(x)\nfig,axes=plt.subplots(1,4,figsize=(11,3))\nfor ax,t in zip(axes,[0,20,80,200]):\n    noisy=e.q_sample(tx,torch.full((len(tx),),t,dtype=torch.long),torch.randn_like(tx),s).numpy()\n    ax.scatter(noisy[:,0],noisy[:,1],s=2,alpha=.25)\n    ax.set(title=f't={t}',xlim=(-4,4),ylim=(-4,4),aspect='equal')\nfig.tight_layout();show(fig);plt.close(fig)")
md('## 5 实际训练三种子\n每个7000次Adam更新。训练t均匀抽样，目标为按坐标平均的无权重噪声MSE，不是精确似然。')
code("results=e.main(Path('notebook_outputs'),steps=7000)\nfor row in results['runs']:\n    print(row['seed'],row['final'],row['history'][-1])")
md('## 6 从纯噪声生成并看完整链\n采样不访问x0；以下中间状态来自同一批反向链。')
code("f=np.load('notebook_outputs/ddpm_seed11_samples.npz')\nfig,axes=plt.subplots(1,5,figsize=(12,3))\nfor ax,t in zip(axes,[200,150,100,50,0]):\n    a=f['path_'+str(t)]\n    ax.scatter(a[:,0],a[:,1],s=2,alpha=.25)\n    ax.set(title=f't={t}',xlim=(-4,4),ylim=(-4,4),aspect='equal')\nfig.tight_layout();show(fig);plt.close(fig)")
md('## 7 端点密度与展示采样\n默认t=1输出均值；terminal_noise=True才从正方差β1的连续高斯端点抽样。两种输出都保存并分别评价。')
code("for row in results['runs']:\n    print('seed',row['seed'],'mean endpoint',row['final']['valid_fraction'],'Gaussian endpoint',row['terminal_gaussian_sample']['valid_fraction'])\nprint('time bins:',results['runs'][0]['time_bins'])\nprint('zero baseline:',results['runs'][0]['zero_predictor_validation_mse'])")
md('## 8 结论检查\n请解释：给定x0的后验高斯为何不代表边缘反向分布必为高斯？无权重MSE为何不能直接叫精确似然？为什么终点barαT非零仍可从标准正态近似采样？不要把二维结果外推为图像基准结论。')
nb=n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}});n.write(nb,R/'experiment.ipynb');print('created',R/'experiment.ipynb')
