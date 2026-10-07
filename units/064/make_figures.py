"""Original project map, measured paired effects, and unfiltered failure examples."""
from pathlib import Path
import os,json
R=Path(__file__).resolve().parent;os.environ.setdefault('MPLCONFIGDIR',str(R/'tmp/mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_path=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if font_path.exists(): font_manager.fontManager.addfont(str(font_path))
from matplotlib.patches import FancyBboxPatch
import numpy as np
plt.rcParams.update({'font.family':['Noto Sans CJK JP','Noto Sans CJK SC','DejaVu Sans'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170,'axes.unicode_minus':False})
F=R/'figures';F.mkdir(exist_ok=True);C=['#187b91','#d17b34','#8865a9'];N=['恒定β','线性预热','双预算恒定β']
def save(fig,name):fig.savefig(F/name,bbox_inches='tight',facecolor='white');plt.close(fig)
def image_grid(arr,cols=8):
 rows=(len(arr)+cols-1)//cols;out=np.full((rows*9-1,cols*9-1),.5)
 for i,x in enumerate(arr):out[(i//cols)*9:(i//cols)*9+8,(i%cols)*9:(i%cols)*9+8]=x.reshape(8,8)
 return out

def main():
 r=json.loads((R/'outputs/results.json').read_text());data=np.load(R/'data/bars.npz')
 fig,ax=plt.subplots(figsize=(10,3.4));ax.axis('off');ax.set(xlim=(0,10),ylim=(0,3.4))
 labels=['假设\n预热提升留出ELBO','机制猜想\n早期先学信息编码','干预\n只改前500步β','证据\n五配对种子+护栏']
 for i,t in enumerate(labels):
  x=.15+2.5*i;ax.add_patch(FancyBboxPatch((x,1.55),2.1,.85,boxstyle='round,pad=.03',fc='#edf4f6',ec='#8196a4'));ax.text(x+1.05,1.98,t,ha='center',va='center')
  if i<3:ax.annotate('',(x+2.45,1.98),(x+2.13,1.98),arrowprops={'arrowstyle':'->','color':'#187b91'})
 ax.text(5,.88,'可证伪门槛：平均 NELBO 至少降低0.2；至少4/5胜；生成有效率下降不超过0.02',ha='center',fontsize=10)
 ax.text(5,.3,'实测：平均差 +0.0458，0/5胜 → 不支持本任务中的预热收益假设',ha='center',color='#a74d20',fontsize=11)
 save(fig,'01_project_map.png')
 fig,axs=plt.subplots(1,2,figsize=(10,3.5));t=np.arange(1,2001);axs[0].plot(t,np.ones_like(t),c=C[0],label='恒定β=1');axs[0].plot(t,np.minimum(t/500,1),c=C[1],label='预热500步');axs[0].set(xlabel='优化器更新步',ylabel='KL权重 β',ylim=(0,1.1));axs[0].legend()
 axs[1].imshow(image_grid(data['test'][:24]),cmap='viridis',vmin=0,vmax=1);axs[1].axis('off');axs[1].set_title('固定留出集前24例 含真实翻转噪声');fig.tight_layout();save(fig,'02_schedule_data.png')
 fig,axs=plt.subplots(1,2,figsize=(10,3.6))
 for j,arm in enumerate(['constant','warmup','long_constant']):
  row=next(x for x in r['runs'] if x['arm']==arm and x['seed']==11);h=row['history'];axs[0].plot([x['step'] for x in h],[x['nll'] for x in h],c=C[j],label=N[j]);axs[1].plot([x['step'] for x in h],[x['kl'] for x in h],c=C[j],label=N[j])
 for ax in axs:ax.set(xlabel='优化器更新步');ax.legend(fontsize=9);ax.grid(alpha=.15)
 axs[0].set(ylabel='小批量重建NLL nats/样本',title='种子11 随机批次轨迹');axs[1].set(ylabel='小批量KL nats/样本',title='不能把变化β的总目标直接横向比较');fig.tight_layout();save(fig,'03_training.png')
 fig,axs=plt.subplots(1,2,figsize=(10,3.6));s=[x['seed'] for x in r['paired']]
 for j,key in enumerate(['warmup_minus_constant','long_minus_constant']):axs[0].plot(s,[x[key] for x in r['paired']],marker='o',c=C[j+1],label=['预热−恒定','双预算−恒定'][j])
 axs[0].axhline(0,c='gray',lw=.8);axs[0].axhline(-.2,c='red',ls=':',label='预设收益门槛−0.2');axs[0].set(xlabel='训练种子',ylabel='测试NELBO差 越低越好');axs[0].legend(fontsize=8)
 for j,arm in enumerate(['constant','warmup','long_constant']):
  rows=[x for x in r['runs'] if x['arm']==arm];axs[1].scatter([j]*5,[x['sample']['template_valid_fraction'] for x in rows],s=50,c=C[j]);axs[1].plot([j-.15,j+.15],[np.mean([x['sample']['template_valid_fraction'] for x in rows])]*2,c='black')
 axs[1].axhline(r['reference_sample_metrics']['template_valid_fraction'],c='#4c8b55',ls=':',label='独立真实测试集');axs[1].set(xticks=range(3),xticklabels=N,ylabel='模板有效比例 ≤4像素差',ylim=(0,1));axs[1].legend(fontsize=8);fig.tight_layout();save(fig,'04_paired_results.png')
 fig,axs=plt.subplots(2,3,figsize=(10,5))
 for j,arm in enumerate(['constant','warmup','long_constant']):
  a=np.load(R/f'outputs/{arm}_seed11_arrays.npz')
  for i,key in enumerate(['sample_probabilities','samples']):axs[i,j].imshow(image_grid(a[key][:24]),cmap='viridis',vmin=0,vmax=1);axs[i,j].axis('off');axs[i,j].set_title(N[j]+(' 解码概率' if i==0 else ' 伯努利抽样'))
 fig.tight_layout();save(fig,'05_prior_samples.png')
 fig,axs=plt.subplots(3,1,figsize=(10,4.2));a=np.load(R/'outputs/warmup_seed11_arrays.npz');idx=a['failure_indices'][:8]
 for ax,x,title in zip(axs,[data['test'][idx],a['mean_reconstruction'][idx],np.abs(data['test'][idx]-a['mean_reconstruction'][idx])],['预热种子11 测试NELBO最高的8例','后验均值解码概率 不是先验生成','绝对像素误差']):ax.imshow(image_grid(x),cmap='viridis',vmin=0,vmax=1);ax.axis('off');ax.set_title(title,fontsize=10)
 fig.tight_layout();save(fig,'06_failure_inspection.png')
if __name__=='__main__':main()
