from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/mpl'))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10})
F=ROOT/'figures';F.mkdir(exist_ok=True);d=np.load(ROOT/'outputs/data.npz');p=np.load(ROOT/'outputs/plot_data.npz');r=json.loads((ROOT/'outputs/results.json').read_text())['results']
def save(name):plt.savefig(F/name,dpi=170,bbox_inches='tight');plt.close()
fig,axes=plt.subplots(1,2,figsize=(9,3));axes[0].scatter(d['x_train'][:,0],d['x_train'][:,1],c=d['y_train'],cmap='tab10',s=8);axes[0].set(xlabel='signal 1',ylabel='signal 2',title='象限标签对应信号角度');axes[1].scatter(d['x_train'][:,4],d['x_train'][:,5],c=d['y_train'],cmap='tab10',s=8);axes[1].set(xlabel='nuisance 1',ylabel='nuisance 2',title='干扰与类别独立');save('01_data.png')
fig,axes=plt.subplots(1,4,figsize=(11,3))
for ax,m in zip(axes,['random','contrastive_good','contrastive_bad','supervised']):
 z=p[m+'_5901_embedding'];ax.scatter(z[:,0],z[:,1],c=d['y_test'],cmap='tab10',s=8);ax.set_title(m,fontsize=9);ax.set_xlabel('h1');ax.set_ylabel('h2')
save('02_embeddings.png')
plt.figure(figsize=(8,2.3))
for i,m in enumerate(['random','contrastive_good','contrastive_bad','supervised']):
 vals=[x['test_accuracy'] for x in r if x['method']==m];plt.bar(i,np.mean(vals),alpha=.65);plt.scatter([i]*3,vals,c='black',s=25)
plt.xticks(range(4),['随机','保语义对比','破语义对比','监督预训练']);plt.axhline(.25,ls='--',c='gray');plt.ylim(0,1.08);plt.ylabel('冻结探测测试准确率');save('03_probe.png')
plt.figure(figsize=(8,3))
for m in ['contrastive_good','contrastive_bad']:
 for seed in [5901,5902,5903]:plt.plot(p[f'{m}_{seed}_loss'],alpha=.7,label=f'{m} {seed}')
plt.legend(fontsize=7,ncol=2);plt.xlabel('优化步');plt.ylabel('InfoNCE');save('04_losses.png')
fig,ax=plt.subplots(figsize=(9,3));ax.axis('off')
for x,y,t in [(.08,.5,'同一原始输入 x'),(.34,.8,'视图 v1'),(.34,.2,'视图 v2'),(.60,.8,'共享编码器 f'),(.60,.2,'共享编码器 f'),(.86,.5,'投影头 g\n对比目标')]:
 ax.text(x,y,t,ha='center',va='center',bbox=dict(boxstyle='round',fc='#e9f1f8',ec='#37607c'),transform=ax.transAxes)
for a,b in [((.16,.55),(.27,.77)),((.16,.45),(.27,.23)),((.41,.8),(.5,.8)),((.41,.2),(.5,.2)),((.7,.8),(.80,.56)),((.7,.2),(.80,.44))]:ax.annotate('',xy=b,xytext=a,xycoords='axes fraction',arrowprops={'arrowstyle':'->'})
save('05_views.png')
plt.figure(figsize=(7,3));temps=np.linspace(.08,2,100)
for gap in [0,.5,1]:plt.plot(temps,np.log1p(2*np.exp(-gap/temps)),label=f'positive gap={gap}')
plt.xlabel('温度');plt.ylabel('三候选交叉熵');plt.legend();save('06_temperature.png')
