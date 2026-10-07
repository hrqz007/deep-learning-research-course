"""Create a teaching notebook which genuinely retrains all four GAN runs."""
from pathlib import Path
import nbformat as n
R=Path(__file__).resolve().parent
cells=[]
def md(s):cells.append(n.v4.new_markdown_cell(s))
def code(s):cells.append(n.v4.new_code_cell(s))
md('# DL061 对抗生成与博弈训练\n\n从新内核顺序运行。这个Notebook会真实训练三个基准和一个压力运行，不依赖隐藏变量或只展示缓存输出。教材固定结果位于outputs；本次重跑写入notebook_outputs。')
code("from pathlib import Path\nimport sys, json, numpy as np, torch\nimport experiment as e\nfrom io import BytesIO\nfrom IPython.display import Image, display\ndef show(fig):\n    buffer = BytesIO()\n    fig.savefig(buffer, format='png', dpi=130, bbox_inches='tight')\n    display(Image(data=buffer.getvalue()))\nprint({'python':sys.version.split()[0], 'torch':torch.__version__, 'unit':Path.cwd().name})\ntorch.set_num_threads(1)")
md('## 1 先手算损失\n真假两组均值相加，所以D=0.5时损失为2log2。G的非饱和损失则为log2。')
code("p=np.array([.8,.6]); q=np.array([.2,.4])\nprint('D batch loss:',-np.log(p).mean()-np.log(1-q).mean())\nprint('D .5 reference:',2*np.log(2),'G .5 reference:',np.log(2))")
md('## 2 行为测试\n检查更新隔离、指标分母、极端logit和可复现性。测试通过不保证收敛，只排除已覆盖的程序错误。')
code("import unittest, test_experiment\nsuite=unittest.defaultTestLoader.loadTestsFromModule(test_experiment)\nresult=unittest.TextTestRunner(verbosity=1).run(suite)\nif not result.wasSuccessful(): raise RuntimeError('unit tests failed')")
md('## 3 原创训练数据与评价器负对照\n八种模式标签只用于评价，不进入GAN训练。常数样本是人为构造，不能作为真实训练坍塌的证据。')
code("x,labels=e.make_data()\nprint('data:',x.shape,'reference metric:',e.mode_metrics(x))\nconstant=np.repeat(e.centers()[:1],4096,axis=0)\nprint('FORCED metric control:',e.mode_metrics(constant))")
md('## 4 实际重新训练\n基准3000轮、三个种子；压力2200轮、种子17。每100轮保存固定噪声的输出，评估随机序列与训练分开。')
code("results=e.main(Path('notebook_outputs'),steps=3000,stress_steps=2200)\nfor run in results['runs']: print(run['seed'],run['final']['coverage'],run['final']['valid_fraction'])")
md('## 5 区分失败快照与最终状态\n诊断规则公开选择有效比例至少0.5的低覆盖快照；最终状态单独报告。')
code("s=results['stress']\nprint('diagnostic:',s['diagnostic'])\nprint('final:',s['final'])\nprint('configuration:',s['lr_g'],s['lr_d'],s['g_steps_per_d'])")
code("import matplotlib.pyplot as plt\nsamples=np.load('notebook_outputs/stress_seed17_samples.npz')\nfig,axes=plt.subplots(1,2,figsize=(8,3.7))\nfor ax,step in zip(axes,[s['diagnostic']['step'],s['steps']]):\n    a=samples[str(step)]\n    ax.scatter(x[:,0],x[:,1],s=2,alpha=.1,color='gray')\n    ax.scatter(a[:,0],a[:,1],s=3,alpha=.3)\n    ax.set(title=f'Stress step {step}',xlim=(-3,3),ylim=(-3,3),aspect='equal')\nfig.tight_layout();show(fig);plt.close(fig)")
md('## 6 理论的数值核对\n使用两格分布计算JS与理想目标。它不等于实际非饱和训练日志中的G损失。')
code("p=np.array([.5,.5]);q=np.array([1.,0.]);m=(p+q)/2\nklp=(p*np.log(p/m)).sum();klq=np.log(1/m[0]);js=(klp+klq)/2\nprint({'KL_p_m':klp,'KL_q_m':klq,'JS':js,'ideal_V':-np.log(4)+2*js})")
md('## 7 写出可支持的结论\n请解释8/8覆盖为何不足以证明分布匹配；0.5判别概率为何不足以证明成功；压力配置为何不能单独归因某一个超参数。完整推导、实验要求与详细答案见同目录三份PDF。')
nb=n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}});n.write(nb,R/'experiment.ipynb');print('created',R/'experiment.ipynb')
