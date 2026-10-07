"""Recreate the executable teaching Notebook. Run execute_notebook.py afterwards."""
from pathlib import Path
import nbformat as n
R=Path(__file__).resolve().parent
cells=[]
cells.append(n.v4.new_markdown_cell('# DL063 生成模型评价与风险\n\n此Notebook实际重训VAE与GAN各三种子。所有模型采用固定终点；控制分布明确标记。教学结果另存notebook_outputs，避免覆盖原outputs。'))
cells.append(n.v4.new_code_cell("from pathlib import Path\nimport sys, json, numpy as np, torch\nimport experiment as e\nimport matplotlib.pyplot as plt\nfrom io import BytesIO\nfrom IPython.display import Image, display\ndef show(fig):\n    b=BytesIO(); fig.savefig(b,format='png',dpi=130,bbox_inches='tight')\n    display(Image(data=b.getvalue())); plt.close(fig)\ntorch.set_num_threads(1)\nprint({'python':sys.version.split()[0],'torch':torch.__version__,'unit':Path.cwd().name})"))
cells.append(n.v4.new_markdown_cell('## 1 指标的分母与盲点\n质量是全部生成点中落入有效域的比例。覆盖要求某模式有效计数至少占全部样本1%。坐标均值协方差相同并不意味着模式相同。'))
cells.append(n.v4.new_code_cell("x=e.centers().astype(float)\na=np.pi/8;rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])\ny=x@rot.T\nprint('rotated quality:',e.mode_metrics(y))\nprint('xy Gaussian feature distance:',e.gaussian_feature_distance(x,y))\nprint('angle features:',e.gaussian_feature_distance(x,y,'radial_angle'))"))
cells.append(n.v4.new_markdown_cell('## 2 普通单元测试\n同一文件还应在命令行使用python -O -m unittest -v test_experiment.py运行，确保验证不依赖assert语句。'))
cells.append(n.v4.new_code_cell("import unittest, test_experiment\nr=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(test_experiment))\nif not r.wasSuccessful(): raise RuntimeError('Tests failed')"))
cells.append(n.v4.new_markdown_cell('## 3 完整真实训练\n三种子×两家族×4000轮。匹配主模型更新次数和真实批次数；GAN另含判别器更新，因此不声称FLOPs相等。'))
cells.append(n.v4.new_code_cell("results=e.main(Path('notebook_outputs'),steps=4000)\nfor row in results['runs']:\n    print(row['family'],row['seed'],row['metrics']['valid_fraction'],row['feature_distance']['xy'])"))
cells.append(n.v4.new_markdown_cell('## 4 看全部样本\n以下固定种子11，画出4096点，不筛选漂亮点。VAE保留条件观测噪声。'))
cells.append(n.v4.new_code_cell("fig,axes=plt.subplots(1,2,figsize=(9,4))\nfor ax,family in zip(axes,['vae','gan']):\n    x=np.load(f'notebook_outputs/{family}_seed11_samples.npz')['samples']\n    ax.scatter(x[:,0],x[:,1],s=2,alpha=.25)\n    ax.set(title=family.upper()+' seed11',xlim=(-3,3),ylim=(-3,3),aspect='equal')\nfig.tight_layout();show(fig)"))
cells.append(n.v4.new_markdown_cell('## 5 样本量改变估计值\n两组来自同一个真实分布的独立样本，总体距离应为0；有限样本插件估计通常为正。误差线表示20次重复的标准差。'))
cells.append(n.v4.new_code_cell("rows=results['sample_size_audit']\nfig,ax=plt.subplots(figsize=(7,3))\nax.errorbar([r['n'] for r in rows],[r['mean'] for r in rows],yerr=[r['std'] for r in rows],marker='o',capsize=4)\nax.set(xscale='log',xlabel='Sample count per group',ylabel='Gaussian feature distance')\nshow(fig)"))
cells.append(n.v4.new_markdown_cell('## 6 最近邻记忆审计\n两个参考集大小相同。复制器质量和覆盖可以满分，但训练精确匹配率为1；这只是检测已知复制的阳性控制，不是证明网络具备或缺乏所有隐私风险。'))
cells.append(n.v4.new_code_cell("for key,row in results['controls'].items(): print(key,row['audit'])\nfor row in results['runs']: print(row['family'],row['seed'],row['audit'])"))
cells.append(n.v4.new_markdown_cell('## 7 检查点复生\n受限加载自己生成的权重，固定采样随机数，应逐元素重现保存样本。'))
cells.append(n.v4.new_code_cell("for row in results['runs']:\n    family,seed=row['family'],row['seed']\n    c=torch.load(f'notebook_outputs/{family}_seed{seed}.pt',weights_only=True)\n    m=e.VAE() if family=='vae' else e.Generator();m.load_state_dict(c['state_dict']);m.eval()\n    sample,means=e.sample_model(m,family,c['sampling_seed'])\n    stored=np.load(f'notebook_outputs/{family}_seed{seed}_samples.npz')\n    np.testing.assert_array_equal(sample,stored['samples']);np.testing.assert_array_equal(means,stored['means'])\nprint('All six checkpoints reproduce both saved arrays exactly.')"))
cells.append(n.v4.new_markdown_cell('## 结论练习\n为什么本次坐标高斯距离偏好VAE，但有效比例偏好GAN？给出特征不充分、有限样本和模型差异的分别解释。该实验不能回答条件一致性、真实图像语义质量或严格隐私保证。'))
nb=n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','name':'python3','language':'python'}})
n.write(nb,R/'experiment.ipynb')
