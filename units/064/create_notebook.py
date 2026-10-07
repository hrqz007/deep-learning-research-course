"""Recreate the executable teaching Notebook. Run execute_notebook.py afterwards."""
from pathlib import Path
import nbformat as n
R=Path(__file__).resolve().parent
cells=[]
cells.append(n.v4.new_markdown_cell('# DL064 表示或生成研究项目\n\n研究问题：在相同2000次更新下，VAE前500步线性KL预热能否改善留出NELBO且不牺牲生成质量？五配对种子、固定终点、双预算对照，真实重训。'))
cells.append(n.v4.new_code_cell("from pathlib import Path\nimport sys, json, numpy as np, torch\nimport experiment as e\nimport matplotlib.pyplot as plt\nfrom io import BytesIO\nfrom IPython.display import Image, display\ndef show(fig):\n    b=BytesIO(); fig.savefig(b,format='png',dpi=130,bbox_inches='tight')\n    display(Image(data=b.getvalue())); plt.close(fig)\ntorch.set_num_threads(1)\nprint({'python':sys.version.split()[0],'torch':torch.__version__,'unit':Path.cwd().name})"))
cells.append(n.v4.new_markdown_cell('## 1 固定数据和可证伪门槛\n平均NELBO差≤−0.2 nats，至少4/5种子优于基线，生成模板有效率平均差≥−0.02。三个条件都满足才支持本次假设。'))
cells.append(n.v4.new_code_cell("data=e.make_data()\nprint({k:v.shape for k,v in data.items()})\nprint('beta warmup:',[e.beta_at(s,'warmup') for s in [1,100,500,1000]])\nprint('test reference:',e.sample_metrics(data['test']))"))
cells.append(n.v4.new_markdown_cell('## 2 概率目标单元测试\n像素先求和、批次后平均。KL与重建项都用每样本nats；β=1时负ELBO可比较。'))
cells.append(n.v4.new_code_cell("import unittest, test_experiment\nr=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(test_experiment))\nif not r.wasSuccessful(): raise RuntimeError('Tests failed')"))
cells.append(n.v4.new_markdown_cell('## 3 完整研究实验\n15个模型：恒定β、预热各2000步，双预算恒定β4000步，各5种子。评估采用32次共享噪声的Monte Carlo估计。'))
cells.append(n.v4.new_code_cell("results=e.main(Path('notebook_outputs'),steps=2000,warmup=500)\nprint('Predeclared decision:',results['decision'])\nprint('Independent-pixel baseline:',results['independent_pixel_baseline_nll'])"))
cells.append(n.v4.new_markdown_cell('## 4 配对差而非挑最好种子\n负值意味着比同种子恒定β基线更好。五种子只描述固定数据集的优化随机性。'))
cells.append(n.v4.new_code_cell("rows=results['paired'];fig,ax=plt.subplots(figsize=(8,3))\nfor key,label in [('warmup_minus_constant','Warmup - constant'),('long_minus_constant','Long - constant')]:\n    ax.plot([r['seed'] for r in rows],[r[key] for r in rows],marker='o',label=label)\nax.axhline(0,color='gray');ax.axhline(-.2,color='red',ls=':')\nax.set(xlabel='Training seed',ylabel='Test negative ELBO difference');ax.legend();show(fig)"))
cells.append(n.v4.new_markdown_cell('## 5 概率图和真实伯努利抽样\n图中固定seed11前8例；不依据观感选择。第一排为概率，第二排为实际二值采样。'))
cells.append(n.v4.new_code_cell("a=np.load('notebook_outputs/warmup_seed11_arrays.npz')\nfig,axes=plt.subplots(2,8,figsize=(10,3))\nfor i in range(8):\n    axes[0,i].imshow(a['sample_probabilities'][i].reshape(8,8),vmin=0,vmax=1,cmap='viridis')\n    axes[1,i].imshow(a['samples'][i].reshape(8,8),vmin=0,vmax=1,cmap='viridis')\nfor ax in axes.ravel():ax.axis('off')\nfig.suptitle('Warmup seed11: decoder probability / actual Bernoulli sample');show(fig)"))
cells.append(n.v4.new_markdown_cell('## 6 失败样本\n按保存的逐例测试NELBO排序，只用于解释，不用于回改本次模型。记录索引后可在相同数据上复查。'))
cells.append(n.v4.new_code_cell("idx=a['failure_indices'][:8]\nprint('Failure indices:',idx.tolist())\nprint('Per-example NELBO:',a['negative_elbo'][idx])\nfig,axes=plt.subplots(2,8,figsize=(10,3))\nfor j,i in enumerate(idx):\n    axes[0,j].imshow(data['test'][i].reshape(8,8),vmin=0,vmax=1,cmap='viridis')\n    axes[1,j].imshow(a['mean_reconstruction'][i].reshape(8,8),vmin=0,vmax=1,cmap='viridis')\nfor ax in axes.ravel():ax.axis('off')\nfig.suptitle('Highest held-out NELBO: original / posterior-mean decoding');show(fig)"))
cells.append(n.v4.new_markdown_cell('## 7 检查点复现\n加载本Notebook实际训练出的15个检查点，重生1024样本与概率。'))
cells.append(n.v4.new_code_cell('for row in results[\'runs\']:\n    key=f"{row[\'arm\']}_seed{row[\'seed\']}"\n    c=torch.load(f\'notebook_outputs/{key}.pt\',weights_only=True)\n    m=e.VAE();m.load_state_dict(c[\'state_dict\']);m.eval()\n    samples,probs=e.sample_model(m,c[\'sampling_seed\'])\n    saved=np.load(f\'notebook_outputs/{key}_arrays.npz\')\n    np.testing.assert_array_equal(samples,saved[\'samples\']);np.testing.assert_array_equal(probs,saved[\'sample_probabilities\'])\nprint(\'All 15 checkpoints reproduce saved samples and probabilities exactly.\')'))
cells.append(n.v4.new_markdown_cell('## 结论\n当前实验不支持预热收益假设；这不是失败的研究交付。请写出机制为何在本模型中可能不成为瓶颈，哪些证据仍然缺失，下一次如何使用新验证设计而不继续消费这份测试集。'))
nb=n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','name':'python3','language':'python'}})
n.write(nb,R/'experiment.ipynb')
