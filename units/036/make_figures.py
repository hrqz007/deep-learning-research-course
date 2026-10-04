"""Rebuild original explanatory figures from hash-guarded teaching results."""
from pathlib import Path
import csv,hashlib,json,os,tempfile
os.environ['MPLCONFIGDIR']=str(Path(tempfile.gettempdir())/'dl036-mpl');os.environ['XDG_CACHE_HOME']=str(Path(tempfile.gettempdir())/'dl036-cache')
Path(os.environ['MPLCONFIGDIR']).mkdir(exist_ok=True);Path(os.environ['XDG_CACHE_HOME']).mkdir(exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parent
EXPECTED={'data/samples.json': '700c21b235136c86c6ca4f7f32ed87976028fd9334f2de169d02a375c6d163db', 'data/hand-example.json': 'c2e88058701be66ea72f1df58355950fc0e24fc7f40287db2de14f78c42d529e', 'data/config.json': '0182da5e95f6cf094ca7b7c9c8dd0214160bb9032034e824051a1e5177dc5d74', 'outputs/batch-sampling.csv': '03dad447e6699b988a4a4377ed383fa42444248b0d2b9e1072e6f56102e7e9e9', 'outputs/training.csv': 'ff7cfd0285a18728b376c1703504c542c403837f2c7d079383d4ac6c8ac804cf', 'outputs/summary.json': 'de2b85e4127ba8910ee12f5dca40ba1f266bde3cf8e9cca9e6a906632d59806c', 'outputs/state-modes.json': 'd1f114499232cfa43c1abe47d4974cdcd374bb03a86f1a1048b8f256b17988b9', 'outputs/batch-sensitivity.json': '682087dab7d7659e96bcd72e5ca72c0e7dcd0d9c8a8a4b82e283b631b8349b19', 'outputs/microbatch.json': '2a5a21778d1a54991b2175f0c421b63b04f76efb033c8c2f2ca527095c62a8ab', 'outputs/axes.json': '3cf80cca1c63a2b0ab68fed161a08e445e472da7e1bd976c1083de4b66607d3e', 'outputs/training-detail.json': '2e449f3331b053318cfcb46618aea463512adb4d061de33244da9e1dd945d682', 'outputs/hand-trace.json': 'a43057fb70937a9eb55be882fbd24b52d5a0515fb35c2093e14295c1ff0978d3', 'data/figure-input-sha256.json': '30605334eb666ec6dd3502d00f0ab66ed01fa6236be2f9c3bfd452efb2394c5f'}
for name,sha in EXPECTED.items():
 if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Teaching input/report changed: '+name)
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(font);plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
BLUE='#2878b5';ORANGE='#e07a26';GREEN='#2b9775';RED='#c34454';OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
def save(name):plt.savefig(OUT/name,bbox_inches='tight',facecolor='white');plt.close()
def get(name):return json.loads((ROOT/'outputs'/name).read_text())
def heat(ax,a,title):
 a=np.asarray(a);im=ax.imshow(a,cmap='RdBu_r',vmin=-max(np.max(abs(a)),.01),vmax=max(np.max(abs(a)),.01));ax.set_title(title);ax.set_xticks(range(a.shape[1]));ax.set_yticks(range(a.shape[0]));
 for i in range(a.shape[0]):
  for j in range(a.shape[1]):ax.text(j,i,f'{a[i,j]:.3g}',ha='center',va='center',color='white' if abs(a[i,j])>.6*np.max(abs(a)) else '#17202a')
 return im
fig,axs=plt.subplots(1,3,figsize=(10.2,3));arr=np.array([[0,2],[1,4],[2,6],[4,-2],[5,0],[6,2]])
for ax,title,group in zip(axs,['输入：两个样本，每个3个位置','BN：每列跨样本及位置','LN(D)：每行跨两个特征'],[0,1,2]):
 heat(ax,arr,title);ax.set_xticks([0,1],['D0','D1']);ax.set_yticks(range(6),['N0 T0','N0 T1','N0 T2','N1 T0','N1 T1','N1 T2']);
 if group==1:
  for j in range(2):ax.add_patch(plt.Rectangle((j-.45,-.45),.9,5.9,fill=False,lw=2.5,edgecolor=[ORANGE,GREEN][j]))
 if group==2:
  for i in range(6):ax.add_patch(plt.Rectangle((-.45,i-.43),1.9,.86,fill=False,lw=2,edgecolor=ORANGE))
fig.tight_layout();save('01_axes.png')
v=np.logspace(-7,2,300);fig,axs=plt.subplots(1,2,figsize=(9.4,3.1))
for ep,c in [(1e-5,BLUE),(1e-3,ORANGE),(1/3,GREEN)]:axs[0].semilogx(v,v/(v+ep),label=f'eps={ep:.4g}',c=c);axs[1].loglog(v,ep/(v+ep)**1.5,label=f'eps={ep:.4g}',c=c)
axs[0].set(xlabel='组内原方差 v',ylabel='归一化后方差 v/(v+eps)',ylim=(-.03,1.05));axs[1].set(xlabel='组内原方差 v',ylabel='尺度方向 Jacobian 特征值');axs[0].legend();axs[1].legend();fig.tight_layout();save('02_epsilon.png')
fig,ax=plt.subplots(figsize=(10,3.4));ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off');nodes={'Z':(1,2),'均值 mu':(3,3.2),'中心化 C':(4.8,2),'方差 v':(6.5,3.2),'标准差 s':(8.1,3.2),'Q=C/s':(8.1,1.3),'H=gamma Q+beta':(5.4,.45),'预测 P 与 loss':(2.1,.45)}
for text,(x,y) in nodes.items():ax.text(x,y,text,ha='center',va='center',bbox={'boxstyle':'round,pad=.45','fc':'#eef4f8','ec':BLUE})
for a,b in [('Z','均值 mu'),('Z','中心化 C'),('均值 mu','中心化 C'),('中心化 C','方差 v'),('方差 v','标准差 s'),('标准差 s','Q=C/s'),('中心化 C','Q=C/s'),('Q=C/s','H=gamma Q+beta'),('H=gamma Q+beta','预测 P 与 loss')]:
 x,y=nodes[a];xx,yy=nodes[b]
 if a=='H=gamma Q+beta':
  ax.annotate('',xy=(3.15,.45),xytext=(4.25,.45),arrowprops={'arrowstyle':'->','color':BLUE,'shrinkA':0,'shrinkB':0});continue
 ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':BLUE,'shrinkA':30,'shrinkB':34})
ax.text(.2,3.65,'反向必须同时走直接、均值、方差三条路径',color=RED);save('03_graph.png')
f=get('hand-trace.json')['steps'][0]['forward_backward'];fig,axs=plt.subplots(1,2,figsize=(9.4,3.5));heat(axs[0],f['local']['jacobian_q_by_z'][0],'单特征 J：输出行 i 对输入行 k');heat(axs[1],np.asarray(f['chain']['loss_path_input_gradients'])[:,:,0],'每行损失传向三个输入行');axs[0].set(xlabel='输入行 k',ylabel='输出行 i');axs[1].set(xlabel='输入行 k',ylabel='损失项 i')
for ax in axs:ax.set_xticks([0,1,2],[1,2,3]);ax.set_yticks([0,1,2],[1,2,3])
fig.tight_layout();save('04_cross_sample.png')
tr=get('hand-trace.json');fig,axs=plt.subplots(1,2,figsize=(10,3.2));xx=np.arange(13)
for i,s in enumerate(tr['steps']):axs[0].bar(xx+(i-.5)*.35,s['forward_backward']['gradient'],.35,label=f'第{i+1}步',color=[BLUE,ORANGE][i])
axs[0].set_xticks(xx,tr['parameter_order'],rotation=60);axs[0].set_ylabel('完整batch梯度');axs[0].legend();loss=[tr['steps'][0]['forward_backward']['loss']]+[s['next_forward']['loss'] for s in tr['steps']];axs[1].plot(range(3),loss,'o-',c=GREEN);axs[1].set(xlabel='已完成参数更新数',ylabel='同批次训练态半MSE',xticks=[0,1,2]);fig.tight_layout();save('05_gradients.png')
p=get('batch-sensitivity.json')['peer_change'];fig,ax=plt.subplots(figsize=(8.6,3));xx=np.arange(4);a=p['BN_before']+p['LN_before'];b=p['BN_after']+p['LN_after'];ax.bar(xx-.18,a,.36,color=BLUE,label='原同伴');ax.bar(xx+.18,b,.36,color=ORANGE,label='替换同伴');ax.set_xticks(xx,['BN D0','BN D1','LN D0','LN D1']);ax.set_ylabel('固定样本归一化结果');ax.legend();fig.tight_layout();save('06_peer_dependency.png')
s=get('batch-sensitivity.json')['running_shift'];fig,axs=plt.subplots(1,2,figsize=(9.6,3.1));step=[r['step'] for r in s];axs[0].plot(step,[r['source_mean'] for r in s],c='black',label='生成总体均值');axs[0].plot(step,[r['batch_mean'] for r in s],c=BLUE,alpha=.55,label='当前批均值');axs[0].plot(step,[r['running_mean'] for r in s],c=ORANGE,label='运行均值');axs[0].axvline(20.5,c=RED,ls=':');axs[0].set(xlabel='实际前向批次',ylabel='特征0均值');axs[0].legend(fontsize=8);w=.1*.9**np.arange(30);axs[1].bar(np.arange(30),w,color=ORANGE);axs[1].set(xlabel='距当前多少批',ylabel='momentum .1 的批均值权重');fig.tight_layout();save('07_running_stats.png')
fig,ax=plt.subplots(figsize=(9.2,3));ax.axis('off');rows=[['train + grad','当前batch','是','是'],['train + no_grad','当前batch','是','否'],['eval + grad','running统计','否','是'],['eval + no_grad','running统计','否','否']];tab=ax.table(cellText=rows,colLabels=['模型态与梯度态','归一化使用','更新buffer','建立梯度图'],cellLoc='center',loc='center',colWidths=[.3,.27,.22,.21]);tab.auto_set_font_size(False);tab.set_fontsize(11);tab.scale(1,2.1)
for (row,col),cell in tab.get_celld().items():cell.set_edgecolor('#ced8e1');cell.set_facecolor('#eaf1f7' if row==0 else ('#fafbfc' if row%2 else 'white'))
ax.set_title('默认 track_running_stats=True；存在需梯度输入/参数',pad=12);save('08_modes.png')
rows=list(csv.DictReader((ROOT/'outputs/batch-sampling.csv').open()));fig,axs=plt.subplots(1,2,figsize=(9.4,3.1));Bs=[2,4,8,32]
for ax,key,title in zip(axs,['anchor_channel0','mean_channel0'],['固定锚点0.5的归一化输出','包含固定锚点的批均值']):
 vals=[[float(r[key]) for r in rows if int(r['batch_size'])==B] for B in Bs];ax.boxplot(vals,tick_labels=Bs,patch_artist=True,boxprops={'facecolor':'#bed7ee'});ax.set(xlabel='batch_size，含1个固定锚点',ylabel=title)
fig.tight_layout();save('09_small_batch.png')
details=get('training-detail.json');fig,axs=plt.subplots(1,2,figsize=(9.5,3.1))
for d in details:
 c=BLUE if d['batch_size']==4 else ORANGE;h=d['history'];label=f"B={d['batch_size']}" if d['shuffle_seed']==360 else None;axs[0].plot([x['epoch'] for x in h],[x['validation_half_mse'] for x in h],c=c,alpha=.7,label=label);axs[1].plot([x['steps'] for x in h],[x['validation_half_mse'] for x in h],c=c,alpha=.7,label=label)
axs[0].set(xlabel='epoch，相同样本呈现数',ylabel='正确eval验证半MSE');axs[1].set(xlabel='optimizer更新数',ylabel='正确eval验证半MSE');axs[0].legend();axs[1].legend();fig.tight_layout();save('10_training_budget.png')
rows=list(csv.DictReader((ROOT/'outputs/training.csv').open()));fig,axs=plt.subplots(1,2,figsize=(9.4,2.7));x=np.arange(6)
for i,(key,label,c) in enumerate([('eval_half_mse','正确eval',BLUE),('wrong_train_no_grad_half_mse','误用train+no_grad',ORANGE),('eval_after_contamination_half_mse','污染后再eval',GREEN)]):axs[0].bar(x+(i-1)*.25,[float(r[key]) for r in rows],.25,label=label,color=c)
axs[0].set_xticks(x,[r['batch_size']+'/'+r['shuffle_seed'] for r in rows],rotation=40);axs[0].set_ylabel('同一验证数据半MSE');fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=3,bbox_to_anchor=(.51,1.07),fontsize=9);axs[1].bar(x,[float(r['running_mean_max_change']) for r in rows],color=RED);axs[1].set_xticks(x,[r['batch_size']+'/'+r['shuffle_seed'] for r in rows],rotation=40);axs[1].set_ylabel('运行均值最大变化');fig.tight_layout();save('11_validation_state.png')
print('11 original figures rebuilt')
