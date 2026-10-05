"""Create source Notebook; execution is a separate new-process step."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parent
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
cells=[md('''# 052 注意力的逐项计算
从同一实例的手算到两次同步更新，再检查mask与未来泄漏。Run All会重跑全部12配置，不读取保存图片冒充新图。CPU float64、无dropout，完整Notebook需要Noto Sans CJK系统字体；安装与检查见lab.md。新进程内真实IPython核已用于离线验证；浏览器和socket不在该验证范围。'''),code('''from pathlib import Path
import os,sys,json,io,tempfile,unittest,hashlib,math
root=Path.cwd()
if (root/'units/052').is_dir():root=root/'units/052'
if not (root/'experiment.py').is_file():raise RuntimeError('从课程根或units/052启动Notebook')
sys.path.insert(0,str(root))
import numpy as np,torch
from IPython.display import display,Image
from experiment import attention,reverse,scalar_attention,hand_inputs,objective,hand_ledger,tensor,torch_objective,verify_derivatives,api_probes,leakage_probe,run
import make_figures as figs
import matplotlib.pyplot as plt
import test_experiment
torch.set_num_threads(1)
def require(ok,message):
    if not ok:raise RuntimeError(message)
def show(fig):
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=150,bbox_inches='tight');display(Image(data=buffer.getvalue()));plt.close(fig)
print({'torch':torch.__version__,'numpy':np.__version__,'device':'CPU','dtype':'float64','font':figs.FONT})'''),md('''## 1 独立标量前向与shape
这个例子L=3、S=4、dk=2、dv=3，避免只测试方形矩阵和标量value。mask最后一列无效。'''),code('''rng=np.random.default_rng(5250)
q=rng.normal(size=(2,3,2)).astype(np.float64);k=rng.normal(size=(2,4,2)).astype(np.float64);v=rng.normal(size=(2,4,3)).astype(np.float64)
allow=np.ones((2,3,4),dtype=np.bool_);allow[:,:,3]=False
o,cache=attention(q,k,v,allow);ref=scalar_attention(q,k,v,allow)
np.testing.assert_allclose(o,ref,atol=3e-15,rtol=1e-14)
print('Q,K,V,A,O形状:',q.shape,k.shape,v.shape,cache['a'].shape,o.shape)
print('标量最大差:',np.max(np.abs(o-ref)))
show(figs.flow({}))'''),md('''## 2 同一两样本的完整前向
参数共享；不是每个位置单独训练一套W。每个sample loss是该样本两个位置的half-MSE，总loss是两sample loss平均。'''),code('''hand=hand_ledger();r={'hand':hand}
for state in hand:
    print('state',state['step'],'weights',state['weights'],'loss',state['loss'])
    for b in range(2):
        print('sample',b+1,'Q/K/V',state['q'][b],state['k'][b],state['v'][b])
        print('score',state['score'][b],'A',state['a'][b],'O',state['prediction'][b])
show(figs.matrices(r))'''),md('''## 3 完整Jacobian和全部共享路径
所有反向项按旧参数计算。GX包含Q、K、V三条路径；本例X是给定数据，不更新。'''),code('''for state in hand:
    for key in ['jacobian','go','ga','gs','gq','gk','gv','weight_gradient_per_sample','weight_gradient','input_gradient_paths','input_gradient']:
        print('state',state['step'],key,np.array(state[key]))
show(figs.jacobian(r))
show(figs.gradient(r))'''),md('''## 4 两次同步更新与独立有限差分
PyTorch独立重算三轮前向反向；同步更新后的下一轮必须重算全部QKV。'''),code('''x,y,w=hand_inputs()
for state in hand:
    xt=tensor(x,True);wt=[tensor(z,True) for z in w]
    loss,out=torch_objective(xt,tensor(y),wt);loss.backward()
    np.testing.assert_allclose(out.detach().numpy(),state['prediction'],atol=2e-14)
    for j in range(3):np.testing.assert_allclose(wt[j].grad.numpy(),state['weight_gradient'][j],atol=2e-14)
    print('state',state['step'],'loss',float(loss.detach()),'sample2 position1',float(out[1,0,0].detach()))
    w=[z-.1*t.grad.numpy() for z,t in zip(w,wt)]
deriv=verify_derivatives()
for row in deriv['detail']:print(row)
require(deriv['finite_difference_max_abs']<1e-9,'差分误差过大')
print('最大有限差分误差:',deriv['finite_difference_max_abs'])'''),md('''## 5 mask契约与库实测
有效query无key会报错。无效query的零行是显式约定，其A和为0。不能用nan_to_num隐藏数学未定义行。'''),code('''mask=np.ones((2,3,4),dtype=np.bool_);mask[:,2]=False
try:attention(q,k,v,mask)
except ValueError as error:print('预期拒绝:',error)
else:raise RuntimeError('应该拒绝有效query空支持')
valid=np.ones((2,3),dtype=np.bool_);valid[:,2]=False
zero,c=attention(q,k,v,mask,valid);require(np.isfinite(zero).all() and np.all(zero[:,2]==0),'无效query约定错误')
api=api_probes();print(json.dumps(api,ensure_ascii=False,indent=2))
require(api['sdpa_error']<1e-13 and api['mha_error']<1e-13,'API不对齐')
show(figs.masks({}))'''),md('''## 6 未来信息注入
只更改位置2和3，合法前两个输出应不变。失败是泄漏反证；通过一个样例不是全系统无泄漏证明。'''),code('''leak=leakage_probe();print(json.dumps(leak,ensure_ascii=False,indent=2))
require(leak['causal']['prefix_max_change']==0.,'因果前缀改变')
require(leak['unmasked']['prefix_max_change']>1.,'注入没有暴露预期差异')
show(figs.leakage({'leakage':leak}))'''),md('''## 7 所有12配置完整重跑
先核对固定原始数据SHA。没有测试选种子，也没有删掉合法模型未超过零基线的结果。运行结果逐字节与原结果比较；耗时另记录，不拿它作性能基准。'''),code('''manifest=json.loads((root/'data/manifest.json').read_text())
require(hashlib.sha256((root/'data/sequences.json').read_bytes()).hexdigest()==manifest['sha256'],'数据SHA不一致')
with tempfile.TemporaryDirectory(prefix='dl052-replay-') as folder:
    fresh=run(Path(folder))
    require((Path(folder)/'results.json').read_bytes()==(root/'outputs/results.json').read_bytes(),'结果非原字节复现，先核对版本')
require(len(fresh['runs'])==12,'配置缺失')
for trial in fresh['runs']:print(trial['seed'],trial['alpha'],trial['condition'],'train',trial['train_mse'],'test',trial['test_mse'],'zero',trial['zero_baseline_test_mse'])
show(figs.experiment(fresh))'''),md('''## 8 不调用训练评价函数独立复算全部测试预测
对每条序列逐query点积，再用已保存系数和截距，逐项平方误差。按序列保存误差，不能用聚合数字代替原始证据。'''),code('''data=json.loads((root/'data/sequences.json').read_text())
for trial in fresh['runs']:
    test=np.array(data['values'][str(trial['seed'])]['test'],dtype=np.float64)
    a=np.array(trial['attention'],dtype=np.float64);coef=trial['coefficient'];pred=[]
    for sequence in test:
        pred.append([coef[0]*sum(float(sequence[j])*float(a[t,j]) for j in range(4))+coef[1] for t in range(3)])
    pred=np.array(pred,dtype=np.float64);per=((pred-test[:,1:])**2).mean(1)
    np.testing.assert_allclose(pred,trial['test_predictions'],atol=2e-14,rtol=1e-13)
    np.testing.assert_allclose(per,trial['test_per_sequence_mse'],atol=2e-14,rtol=1e-13)
    np.testing.assert_allclose(per.mean(),trial['test_mse'],atol=2e-14,rtol=1e-13)
print('12×512×3个测试预测独立复算通过；合法因果模型均未胜过零基线')
show(figs.scaling(fresh))'''),md('''## 9 不同权重同一输出
代数非唯一性不等同于某个固定模型可以自由换权重，更不能直接声称权重为因果解释。'''),code('''v=np.array([0.,1.,2.],dtype=np.float64)
for a in [np.array([.4,.2,.4],dtype=np.float64),np.array([.2,.6,.2],dtype=np.float64)]:
    s=np.log(a);p=np.exp(s)/np.exp(s).sum();np.testing.assert_allclose(p,a,atol=1e-15);print('weights',a,'output',float(a@v))
show(figs.nonunique({}))'''),md('''## 10 完整契约测试
这些断言不会在python -O下被移除。当前版本特定的空行行为如果变更，应研究原因，不静默清洗或放宽阈值。'''),code('''suite=unittest.defaultTestLoader.loadTestsFromModule(test_experiment)
result=unittest.TextTestRunner(verbosity=2,stream=sys.stdout).run(suite)
require(result.wasSuccessful(),'契约测试失败')
print('完成全部数值、信息边界与原始结果复算。浏览器/socket/GPU/半精度未测试。')''')]
for i,c in enumerate(cells):c['id']=f'dl052-{i:02d}'
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12.14'}})
nbf.write(nb,ROOT/'experiment.ipynb')
