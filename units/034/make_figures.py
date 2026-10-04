"""Original source-backed figures; fixed report hashes checked before plotting."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
EXPECTED={'data/config.json': '44a32083abe6111dd15cb923e1af6c68b93bed4adbf44dcfe2ca779cf186f8d0', 'data/hand-example.json': 'ab398e1c13115642372e5994c6787ad07a07058cae720a472f592308a0b50c89', 'data/samples.csv': '482391854b087c4d063921bae01af4ed2f11c83f19bb2b19c3a36d7a8874671c', 'outputs/hand-trace.json': '80bbb9e8ef1f55f0d59ed8e9c16a6826032e2f6d9c710b7281eb6a4661eb5e18', 'outputs/mechanisms.json': '8443486a5121c76c39e726b53cdafda0b188ae3ffc71bf044f3e84a5a56c72eb', 'outputs/search-grid.csv': '98cec881d02bc18018c1412b6bf01115c1fdc58d0c7897c6f7e4ddee9ab8d07f', 'outputs/selected-trajectories.json': 'de1afecdf7cb8fa6b38c85ea7ce32f75301fd469a85a6ab0a2936a2375510208', 'outputs/selection.json': '99b668a94ea355e2ab011a05cfbb2fd113fe33c6dec178276dffd591ce949938', 'outputs/state-check.json': '4846650dd4217a86f67f2b6e6bd93433a8f567709724e4e80bc4725873658d69', 'outputs/summary.json': 'bda90706a1a8c5a84bb14292eb17ecb5fdfc4b5072a3fa691011be014d18c47e', 'outputs/test-report.json': 'f50a19c5e11eeb9e9cda5e62c58fb3002fc94197a94b633758285876ed363962'}
for name,digest in EXPECTED.items():
 if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Fixed report changed: '+name)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
import numpy as np
font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if font.exists():fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':11,'axes.unicode_minus':False,'axes.spines.right':False,'axes.spines.top':False,'savefig.dpi':175})
C=['#186b93','#c77723','#298667','#8956a3'];out=ROOT/'figures';out.mkdir(exist_ok=True)
def read(n):return json.loads((ROOT/'outputs'/n).read_text())
def save(f,n):f.savefig(out/n,bbox_inches='tight',facecolor='white');plt.close(f)
trace=read('hand-trace.json');mech=read('mechanisms.json');selection=read('selection.json');histories=read('selected-trajectories.json');test=read('test-report.json')
f,axs=plt.subplots(1,2,figsize=(9,3.5));k=np.arange(1,11);beta=.8;w=(1-beta)*beta**(10-k)
axs[0].bar(k,w,color=C[0],label='未修正权重');axs[0].plot(k,w/(1-beta**10),'o-',color=C[1],label=r'除以 $1-\beta^{10}$');axs[0].set_xlabel('历史梯度来自第 k 步');axs[0].set_ylabel('对第10步均值的权重');axs[0].legend(fontsize=9);axs[0].set_title('最近的梯度权重较大，修正后权重和为1')
t=np.arange(1,31);axs[1].plot(t,1-.8**t,label='m / 恒定梯度，β1=.8',color=C[0]);axs[1].plot(t,1-.9**t,label='v / 恒定梯度平方，β2=.9',color=C[2]);axs[1].axhline(1,color=C[1],ls='--',label='修正后的比例');axs[1].set_xlabel('更新步数 t');axs[1].set_ylabel('比例');axs[1].legend(fontsize=8.5);axs[1].set_title('零初始化的缺口可计算');f.tight_layout();save(f,'01_bias_weights.png')
f,ax=plt.subplots(figsize=(9,3.2));ax.axis('off');ax.set_xlim(0,12);ax.set_ylim(0,4)
blocks=[(1.25,3,'X  2×2\n同一两行样本',C[0]),(4,3,'Z → ReLU H\n2×2',C[0]),(7,3,'P  2×1\n残差 r = P−y',C[0]),(10,3,'L = 均值(r²/2)\n0.0221',C[0]),(10,.9,'dP = r/2\n(−.14,−.05)',C[1]),(7,.9,'dH = dP @ u.T\ndZ = dH * (Z>0)',C[1]),(4,.9,'9个参数梯度\n按样本累加',C[1]),(1.25,.9,'m、v、偏差修正\n更新 θ，再前向',C[2])]
for x,y,txt,col in blocks:ax.text(x,y,txt,ha='center',va='center',fontsize=10,bbox={'boxstyle':'round,pad=.6','fc':col+'16','ec':col,'lw':1})
for xs,xe,y in [(2.25,3,3),(5.05,5.9,3),(8.15,9,3),(8.9,8.1,.9),(5.85,5.05,.9),(2.9,2.3,.9)]:ax.annotate('',xy=(xe,y),xytext=(xs,y),arrowprops={'arrowstyle':'->','color':'#455469','lw':1.5})
ax.annotate('',xy=(10,1.7),xytext=(10,2.3),arrowprops={'arrowstyle':'->','color':C[1],'lw':1.5});ax.set_title('同一计算链只在最后换成Adam，反向传播本身没有换算法',loc='left');save(f,'02_trace_flow.png')
f,axs=plt.subplots(1,2,figsize=(9,3.7));r=trace['steps'][0]['adam'];pos=np.arange(9);names=trace['parameter_order']
axs[0].bar(pos,r['coefficient_on_m_hat'],color=C[0]);axs[0].set_ylabel(r'$\alpha / (\sqrt{\hat v}+\epsilon)$');axs[0].set_title('第1步：乘在修正动量上的系数')
axs[1].bar(pos,-np.array(r['adaptive_delta']),color=C[1]);axs[1].axhline(0,color='#aaa');axs[1].set_ylabel('从 θ 中减去的有符号量');axs[1].set_title(r'第1步：实际更新仍要乘 $\hat m$')
for ax in axs:ax.set_xticks(pos,names,rotation=45);ax.set_xlabel('参数坐标');ax.grid(axis='y',alpha=.2)
f.tight_layout();save(f,'03_coordinate_steps.png')
f,axs=plt.subplots(1,2,figsize=(9,3.5));g=np.logspace(-12,0,300)
for eps,col in [(1e-8,C[0]),(.001,C[1])]:axs[0].semilogx(g,g/(g+eps),color=col,label=f'ε={eps:g}')
axs[0].set_xlabel('第1步正梯度 g');axs[0].set_ylabel('更新量 / α');axs[0].set_title('ε主导时，接近0的梯度不会变成满步');axs[0].legend()
axs[1].loglog(g,g/(g+.001),label='正确：g/(|g|+.001)',color=C[0]);axs[1].loglog(g,g/np.sqrt(g*g+.001),label='改成根号内：g/√(g²+.001)',color=C[3],ls='--');axs[1].set_xlabel('第1步正梯度 g');axs[1].set_ylabel('更新量 / α');axs[1].set_title('同一个数写进根号内，不是等价公式');axs[1].legend(fontsize=8);f.tight_layout();save(f,'04_epsilon.png')
f,axs=plt.subplots(1,2,figsize=(9,3.7))
for name,label,col in [('adam_l2','Adam + L2',C[0]),('adamw','AdamW',C[2])]:
 rows=mech['decay_comparison'][name];points=np.array([[1.,-2.]]+[r['theta_after'] for r in rows]);axs[0].plot(points[:,0],points[:,1],'o-',label=label,color=col)
 for i,p in enumerate(points):axs[0].annotate(str(i),p,xytext=(4,4),textcoords='offset points',color=col)
 mhat=rows[1]['m_hat'];axs[1].bar(np.arange(2)+(-.17 if name=='adam_l2' else .17),mhat,width=.32,label=label,color=col)
axs[0].set_xlabel('θ1');axs[0].set_ylabel('θ2');axs[0].set_title('相同外加数据梯度，不同状态和路径');axs[1].axhline(0,color='#bbb');axs[1].set_xticks([0,1],['坐标1','坐标2']);axs[1].set_ylabel(r'第2步 $\hat m$');axs[1].set_title('L2甚至能改变坐标1的动量符号');
for ax in axs:ax.legend(fontsize=9);ax.grid(alpha=.2)
f.tight_layout();save(f,'05_l2_decoupled.png')
f,axs=plt.subplots(1,2,figsize=(9,3.5))
for row,label,col in zip(mech['pure_decay'],['α=.1','α=.01','先.1后.01'],C):axs[0].plot(range(21),row['path'],label=label,color=col)
axs[0].set_xlabel('提供零梯度的更新步数');axs[0].set_ylabel('θ，初始2');axs[0].set_title('AdamW纯衰减，λ=.2，初始动量0');axs[0].legend(fontsize=9)
rows=mech['none_vs_zero'];axs[1].bar(['第一次后','随后grad=None','随后grad=0'],[rows['none']['before'],rows['none']['after'],rows['zero']['after']],color=[C[0],C[3],C[2]]);axs[1].set_ylim(0,1);axs[1].set_ylabel('参数值');axs[1].set_title('已有动量时，None和数值0含义不同');axs[1].tick_params(axis='x',labelsize=8);f.tight_layout();save(f,'06_decay_and_skip.png')
f,axs=plt.subplots(1,3,figsize=(10,3.8));vals=[r['mean_validation_half_mse'] for r in selection['all_candidates']];lo,hi=min(vals),max(vals)
for ax,fam,title in zip(axs,['sgd_momentum','adam','adamw'],['SGD + momentum .8','Adam','AdamW']):
 rows=[r for r in selection['all_candidates'] if r['family']==fam];v=np.array([r['mean_validation_half_mse'] for r in rows]).reshape(3,3);im=ax.imshow(v,cmap='YlGnBu',vmin=lo,vmax=hi)
 for i in range(3):
  for j in range(3):ax.text(j,i,f'{v[i,j]:.4f}',ha='center',va='center',fontsize=9,color='white' if v[i,j]>(lo+hi)/2 else '#17202b')
 best=next(r for r in selection['selected'] if r['family']==fam);i,j=divmod(best['grid_index'],3);ax.add_patch(plt.Rectangle((j-.46,i-.46),.92,.92,fill=False,ec='#c53932',lw=2))
 ax.set_xticks(range(3),['0','.01','.1']);ax.set_yticks(range(3),[f'{rows[i*3]["lr"]:g}' for i in range(3)]);ax.set_xlabel('weight_decay');ax.set_ylabel('学习率');ax.set_title(title,fontsize=11)
f.subplots_adjust(wspace=.45,bottom=.2,top=.85,right=.9);cax=f.add_axes([.93,.2,.016,.6]);f.colorbar(im,cax=cax,label='平均验证半MSE');save(f,'07_search_grid.png')
f,axs=plt.subplots(1,2,figsize=(9,3.7))
for fam,label,col,ls in [('sgd_momentum','SGD+动量',C[0],'-'),('adam','Adam',C[1],'-'),('adamw','AdamW',C[2],'--')]:
 runs=[r for r in histories if r['family']==fam]
 for ax,key in zip(axs,['train_half_mse','validation_half_mse']):
  values=np.array([[r[key] for r in run['history']] for run in runs]);ax.plot(range(97),values.mean(0),label=label,color=col,ls=ls,lw=2);ax.fill_between(range(97),values.min(0),values.max(0),color=col,alpha=.1)
for ax,title in zip(axs,['训练目标，阴影仅为3种顺序范围','验证目标，选择只用了第96步']):ax.set_title(title,fontsize=10);ax.set_xlabel('optimizer step，每步8行');ax.set_ylabel('平均半MSE');ax.set_yscale('log');ax.legend(fontsize=9);ax.grid(alpha=.2)
f.tight_layout();save(f,'08_selected_curves.png')
f,axs=plt.subplots(1,2,figsize=(9,3.5));families=['sgd_momentum','adam','adamw'];labels=['SGD+动量','Adam','AdamW']
for seed,col in zip([340,341,342],C):axs[0].plot(range(3),[next(r['test_half_mse'] for r in test['rows'] if r['family']==fam and r['seed']==seed) for fam in families],'o-',color=col,label=f'顺序seed {seed}')
axs[0].set_xticks(range(3),labels);axs[0].set_ylabel('冻结后测试半MSE');axs[0].set_title('λ=0入选：Adam和AdamW恰好重合');axs[0].legend(fontsize=8)
s=read('state-check.json');delta=np.array(s['weights_only_theta'])-np.array(s['full_theta']);axs[1].bar(range(9),delta,color=C[3]);axs[1].axhline(0,color='#bbb');axs[1].set_xticks(range(9),trace['parameter_order'],rotation=45);axs[1].set_ylabel('只恢复参数 − 完整恢复');axs[1].set_title('第2步切开，第5步比较：状态不能丢');f.tight_layout();save(f,'09_test_and_state.png')
print('9 original figures rebuilt after11 fixed source/report guards')
