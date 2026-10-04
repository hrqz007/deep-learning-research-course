"""Original explanatory/measurement figures; verify sources before writing outputs."""
from pathlib import Path
import argparse,hashlib,json,gzip,io,os,tempfile
ROOT=Path(__file__).resolve().parent

def make(output=None):
 contract=json.loads((ROOT/'figure-contract.json').read_text())
 for name,sha in contract['sha256'].items():
  path=ROOT/name
  if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('figure source integrity failure: '+name)
 raw=gzip.decompress((ROOT/'outputs/results.json.gz').read_bytes());s=json.loads((ROOT/'outputs/summary.json').read_text())
 if hashlib.sha256(raw).hexdigest()!=s['packed_result']['raw_sha256']:raise ValueError('raw result hash mismatch')
 r=json.loads(raw);calc=json.loads((ROOT/'outputs/calculations.json').read_text())
 import numpy as np
 os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'course-041-mpl'))
 os.environ.setdefault('XDG_CACHE_HOME',str(Path(tempfile.gettempdir())/'course-041-cache'))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib import font_manager
 FONT='DejaVu Sans'
 for path in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf']:
  if Path(path).is_file():font_manager.fontManager.addfont(path);FONT=font_manager.FontProperties(fname=path).get_name()
 plt.rcParams.update({'font.family':[FONT,'DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
 palette=['#334155','#94a3b8','#0f766e','#ea580c','#2563eb','#a855f7','#be123c'];colors=dict(zip(r['protocol']['methods'],palette));short={'scratch':'从零','random_probe':'随机冻结','pretrained_probe':'预训练冻结','finetune_small':'小步微调','finetune_equal':'等步微调','lpft':'探测后微调','rbf_ridge':'RBF岭'};files={}
 def save(fig,name):
  fig.tight_layout(pad=1.25);b=io.BytesIO();fig.savefig(b,format='png',dpi=150,metadata={'Software':'Unit041 original figures'});files[name+'.png']=b.getvalue();plt.close(fig)
 def group(task,n,seed=11):return next(g for g in r['target_groups'] if g['task']==task and g['n']==n and g['seed']==seed)
 def run(g,m):return next(a for a in g['runs'] if a['method']==m)
 def candidate(g,m):a=run(g,m);return a['candidates'][a['selected']]
 def predict(t,x):t=np.array(t);return t[3]*np.tanh(x@t[:2]+t[2])+t[4]
 def label(ax,x,y,txt,color='#e2e8f0',w=.21):ax.text(x,y,txt,ha='center',va='center',bbox=dict(boxstyle='round,pad=.6',facecolor=color,edgecolor='white'),transform=ax.transAxes)
 fig,ax=plt.subplots(figsize=(10,3.5));ax.axis('off')
 for y,text,bg in [(.8,'冻结表示','#cbd5e1'),(.5,'全量微调','#bfdbfe'),(.2,'从零训练','#fed7aa')]:
  for x,txt,col in [(.08,'目标输入',None),(.34,text,bg),(.62,'目标任务头','#bbf7d0'),(.88,'目标损失',None)]:label(ax,x,y,txt,col or '#f1f5f9')
  for a,b in [(.15,.24),(.44,.53),(.72,.80)]:ax.annotate('',(b,y),(a,y),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#334155'})
 ax.set_title('是否更新表示参数，与是否计算输入梯度是两个决定',pad=15);save(fig,'01_transfer_paths')
 fig,axs=plt.subplots(1,2,figsize=(10,3.5))
 for ax in axs:ax.set(xlim=(-2.4,2.5),ylim=(-2.4,2.4),xlabel='x₁',ylabel='x₂');ax.grid(alpha=.15)
 from matplotlib.patches import Rectangle
 axs[0].add_patch(Rectangle((-2,-2),4,4,fill=False,lw=2,label='源输入支持',edgecolor='#64748b'));axs[0].add_patch(Rectangle((-1.5,-2),3.5,4,fill=False,lw=2,ls='--',label='目标输入支持',edgecolor='#2563eb'));axs[0].legend(fontsize=9);axs[0].set_title('域变化：输入边缘分布改变')
 for vec,col,txt in [((1.5,0),'#0f766e','相关任务依赖 x₁'),((0,1.5),'#ea580c','正交任务依赖 x₂')]:axs[1].arrow(0,0,*vec,color=col,width=.035,head_width=.2,length_includes_head=True,label=txt)
 axs[1].legend(loc='lower right',fontsize=9);axs[1].set_title('任务变化：同一目标 x，标签规则改变');save(fig,'02_domain_and_task')
 fig,axs=plt.subplots(1,2,figsize=(10,3.4));xx=np.linspace(-2,2,300)
 axs[0].plot(xx,np.tanh(1.2*xx),color='#0f766e');axs[0].set(xlabel='保留的 x₁',ylabel='h = tanh(1.2 x₁)',title='源任务鼓励保留第一个方向')
 axs[1].scatter([.5,.5],[-1,1],c=['#2563eb','#ea580c'],s=120);axs[1].annotate('相同表示，两个目标值',(.5,0),(.02,0),arrowprops={'arrowstyle':'->'});axs[1].set(xlim=(-.1,1.1),ylim=(-1.4,1.4),xlabel='冻结表示 h',ylabel='正交目标 y',title='任务需要被压缩掉的方向');save(fig,'03_information_loss')
 h=calc['first'];fig,axs=plt.subplots(1,2,figsize=(10,3.5));v=np.array(h['parameter_contributions']);im=axs[0].imshow(v,cmap='coolwarm',vmin=-.4,vmax=.4,aspect='auto');axs[0].set_xticks(range(5),['a₁','a₂','c','v','b']);axs[0].set_yticks([0,1],['样本1','样本2']);axs[0].set_title('每个样本对平均损失梯度的贡献')
 for i in range(2):
  for j in range(5):axs[0].text(j,i,f'{v[i,j]:.3f}',ha='center',va='center',color='black')
 axs[1].bar(range(5),h['gradient'],color=['#2563eb']*3+['#0f766e']*2);axs[1].axhline(0,color='black',lw=.6);axs[1].set_xticks(range(5),['a₁','a₂','c','v','b']);axs[1].set_title('先累加，后按组同步更新');axs[1].set_ylabel('∂L/∂参数');save(fig,'04_hand_gradient_contributions')
 fig,axs=plt.subplots(1,2,figsize=(10,3.5));names=['旧参数','只更新任务头','分组更新全部'];traces=[h,calc['frozen_next'],calc['grouped_next']]
 for i,t in enumerate(traces):axs[0].plot([1,2],t['prediction'],'o-',label=names[i])
 axs[0].scatter([1,2],[1,-.4],marker='x',s=90,color='black',label='标签');axs[0].set_xticks([1,2]);axs[0].set(xlabel='同一批样本',ylabel='预测',title='每个方案都从头重新前向');axs[0].legend(fontsize=8)
 axs[1].bar(names,[t['risk'] for t in traces],color=['#64748b','#0f766e','#2563eb']);axs[1].set(ylabel='平均半平方误差',title='一步后损失，只是当前两条样本');save(fig,'05_hand_next_forward')
 fig,ax=plt.subplots(figsize=(9,3.4));mask=np.array([[1,0,1],[0,0,1],[0,0,1],[1,1,1]]);ax.imshow(mask,cmap='Blues',vmin=0,vmax=1,aspect='auto');ax.set_xticks(range(3),['输入 x 梯度','表示参数梯度','任务头梯度']);ax.set_yticks(range(4),['参数 requires_grad=False','表示 detach()','表示放进 no_grad','仅 module.eval()'])
 for i in range(4):
  for j in range(3):ax.text(j,i,'有' if mask[i,j] else '无',ha='center',va='center',color='white' if mask[i,j] else '#334155')
 ax.set_title('输入 x 需要梯度且任务头可训练时的实测路径');save(fig,'06_freezing_is_not_detach')
 fig,ax=plt.subplots(figsize=(10,3.5));ax.axis('off')
 for x,y,txt,color in [(.18,.78,'源训练512条\n600步 × 3种子','#cbd5e1'),(.57,.78,'目标训练16或64条\n每方法3候选','#bfdbfe'),(.57,.43,'目标验证128条\n只选最终候选','#bbf7d0'),(.88,.2,'目标测试512条\n选择冻结后评价','#fed7aa')]:label(ax,x,y,txt,color)
 for a,b in [((.30,.78),(.43,.78)),((.57,.65),(.57,.55)),((.68,.35),(.78,.24))]:ax.annotate('',b,a,xycoords='axes fraction',arrowprops={'arrowstyle':'->','lw':2})
 ax.text(.08,.16,'输入字节可先核验哈希；测试标签在所有选择后才解析。\n验证标签也计入标签预算：16+128 或 64+128。',transform=ax.transAxes);save(fig,'07_data_boundary')
 fig,axs=plt.subplots(2,2,figsize=(11,6.5),sharey=True)
 for ax,(task,n) in zip(axs.flat,[(a,b) for a in ('related','orthogonal') for b in (16,64)]):
  for i,m in enumerate(r['protocol']['methods']):
   vals=[z['test_mse'] for z in r['metrics'] if z['task']==task and z['n']==n and z['method']==m];ax.scatter(np.full(3,i)+[-.1,0,.1],vals,color=colors[m],s=23);ax.plot([i-.22,i+.22],[np.mean(vals)]*2,color=colors[m],lw=2)
  ax.set_yscale('log');ax.set_xticks(range(7),[short[m] for m in r['protocol']['methods']],rotation=30,ha='right',fontsize=8);ax.set_title(('相关' if task=='related' else '正交')+f'任务，训练标签 n={n}');ax.set_ylabel('测试 MSE，对数轴');ax.grid(axis='y',alpha=.2)
 save(fig,'08_all_test_results')
 fig,axs=plt.subplots(1,2,figsize=(11,3.8),sharey=True)
 for ax,n in zip(axs,(16,64)):
  for i,m in enumerate(['pretrained_probe','finetune_small','finetune_equal','lpft']):
   for dy,task,mark in [(-.12,'related','o'),(.12,'orthogonal','s')]:
    row=next(z for z in r['summary'] if z['task']==task and z['n']==n and z['method']==m);ax.scatter(row['paired_delta_vs_scratch'],np.full(3,i+dy),color=colors[m],marker=mark,s=35)
  ax.axvline(0,color='#334155',lw=1);ax.set_xscale('symlog',linthresh=.01);ax.set_yticks(range(4),[short[m] for m in ['pretrained_probe','finetune_small','finetune_equal','lpft']]);ax.set(xlabel='迁移MSE − 配对从零MSE',title=f'n={n}；圆点相关，方点正交');ax.grid(axis='x',alpha=.2)
 save(fig,'09_paired_negative_transfer')
 fig,axs=plt.subplots(1,2,figsize=(11,3.7));g=group('orthogonal',64)
 for m in ['scratch','finetune_small','finetune_equal','lpft']:
  hist=candidate(g,m)['history']
  for ax,key in zip(axs,['train_mse','validation_mse']):ax.plot([a['step'] for a in hist],[a[key] for a in hist],label=short[m],color=colors[m]);ax.set_yscale('log')
 for ax,txt in zip(axs,['训练','验证']):ax.set(xlabel='目标 SGD 更新次数',ylabel='MSE',title=f'正交任务 {txt}轨迹；n=64，seed=11');ax.legend(fontsize=8);ax.grid(alpha=.2)
 save(fig,'10_actual_learning_curves')
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 for ax,n in zip(axs,(16,64)):
  g=group('orthogonal',n)
  for m in ['scratch','finetune_small','finetune_equal','lpft']:
   pts=np.array([v['theta'] for v in candidate(g,m)['history']]);ax.plot(pts[:,0],pts[:,1],color=colors[m],label=short[m]);ax.scatter(*pts[0,:2],color=colors[m],marker='o',s=24);ax.scatter(*pts[-1,:2],color=colors[m],marker='x',s=60)
  ax.axvline(0,color='#94a3b8',lw=.8);ax.axhline(0,color='#94a3b8',lw=.8);ax.set(xlabel='a₁',ylabel='a₂',title=f'表示方向如何旋转；n={n}');ax.legend(fontsize=8)
 save(fig,'11_parameter_paths')
 grid=np.linspace(-2,2,81);gx,gy=np.meshgrid(grid,grid);xy=np.column_stack([gx.ravel(),gy.ravel()]);fig,axs=plt.subplots(2,3,figsize=(10.5,6));g=group('orthogonal',64)
 for ax,m in zip(axs.flat,['truth','pretrained_probe','finetune_small','finetune_equal','scratch','rbf_ridge']):
  if m=='truth':p=np.tanh(1.2*xy[:,1])
  elif m=='rbf_ridge':
   centers=np.array([(a,b) for a in np.linspace(-2,2,5) for b in np.linspace(-2,2,5)]);phi=np.column_stack([np.exp(-((xy[:,None]-centers[None])**2).sum(2)/2),np.ones(len(xy))]);p=phi@candidate(g,m)['coefficients']
  else:p=predict(candidate(g,m)['theta'],xy)
  im=ax.imshow(p.reshape(gx.shape),extent=[-2,2,-2,2],origin='lower',cmap='coolwarm',vmin=-1.2,vmax=1.2,aspect='auto');fig.colorbar(im,ax=ax,fraction=.05,pad=.03);ax.axvline(-1.5,color='black',ls=':',lw=.7);ax.set(title='目标无噪声函数' if m=='truth' else short[m],xlabel='x₁',ylabel='x₂')
 fig.suptitle('正交目标预测面；n=64 seed=11；虚线左侧不在目标输入支持内',fontsize=11);save(fig,'12_prediction_surfaces')
 fig,axs=plt.subplots(2,2,figsize=(10,6))
 import csv
 test=list(csv.DictReader((ROOT/'data/target_test.csv').read_text().splitlines()));x=np.array([[float(z['x1']),float(z['x2'])] for z in test])
 for row,task in enumerate(('related','orthogonal')):
  y=np.array([float(z[task]) for z in test]);g=group(task,64)
  for col,m in enumerate(('pretrained_probe','finetune_equal')):
   t=np.array(candidate(g,m)['theta']);hv=np.tanh(x@t[:2]+t[2]);axs[row,col].scatter(hv,y,s=5,alpha=.4,color=colors[m]);axs[row,col].set(xlabel='最终表示 h',ylabel='目标 y',title=('相关' if task=='related' else '正交')+'任务 / '+short[m])
 save(fig,'13_representation_and_target')
 fig,axs=plt.subplots(1,2,figsize=(10,3.7));methods=['scratch','finetune_small','finetune_equal','lpft']
 for ax,task in zip(axs,('related','orthogonal')):
  values=np.array([[c['validation_mse'] for c in run(group(task,64),m)['candidates']] for m in methods]);ax.imshow(np.log10(values),cmap='viridis',aspect='auto',vmin=-2.5,vmax=0)
  for i in range(4):
   for j in range(3):ax.text(j,i,f'{values[i,j]:.4f}',ha='center',va='center',color='black' if values[i,j]>.1 else 'white',fontsize=9)
  ax.set_xticks(range(3),['0.03','0.1','0.3']);ax.set_yticks(range(4),[short[m] for m in methods]);ax.set(xlabel='任务头学习率',title=('相关' if task=='related' else '正交')+'任务验证 MSE；seed=11 n=64')
 save(fig,'14_validation_candidates')
 out=ROOT/'figures' if output is None else Path(output)
 if out.exists() and not out.is_dir():raise ValueError('figure output must be directory')
 # Every figure has been computed and serialized successfully before writes.
 out.mkdir(parents=True,exist_ok=True)
 for name,b in files.items():
  fd,tmp=tempfile.mkstemp(prefix='.'+name,dir=out)
  with os.fdopen(fd,'wb') as f:f.write(b)
  os.replace(tmp,out/name)
 return {k:{'sha256':hashlib.sha256(v).hexdigest(),'bytes':len(v)} for k,v in files.items()}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args();print(json.dumps(make(a.output),indent=2))
