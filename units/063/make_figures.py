"""Original explanatory figures from locally measured results."""
from pathlib import Path
import os,json
R=Path(__file__).resolve().parent;os.environ.setdefault('MPLCONFIGDIR',str(R/'tmp/mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_path=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if font_path.exists(): font_manager.fontManager.addfont(str(font_path))
import numpy as np
from matplotlib.patches import FancyBboxPatch,Circle
plt.rcParams.update({'font.family':['Noto Sans CJK JP','Noto Sans CJK SC','DejaVu Sans'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170,'axes.unicode_minus':False})
F=R/'figures';F.mkdir(exist_ok=True)
C={'vae':'#187b91','gan':'#d17b34','oracle':'#3c8d56','copier':'#9569ad'}
def save(fig,name):fig.savefig(F/name,bbox_inches='tight',facecolor='white');plt.close(fig)
def cloud(ax,x,title):
    ax.scatter(x[:,0],x[:,1],s=2,alpha=.25,c='#187b91',rasterized=True)
    import experiment as e
    for cx,cy in e.centers():ax.add_patch(Circle((cx,cy),.35,fill=False,lw=.7,color='#d17b34'))
    ax.set(xlim=(-2.8,2.8),ylim=(-2.8,2.8),title=title,aspect='equal',xlabel='x1',ylabel='x2')
def main():
 r=json.loads((R/'outputs/results.json').read_text());data=np.load(R/'data/mixture.npz')
 fig,ax=plt.subplots(figsize=(10,3.5));ax.axis('off');ax.set(xlim=(0,10),ylim=(0,3.5))
 blocks=[(.2,1.2,'同一训练集\n2048点'),(3.15,2.0,'VAE\n4000次主更新'),(3.15,.25,'GAN\n4000轮 G+D'),(6.2,1.2,'各4096生成点\n固定终点与种子'),(8.15,1.2,'多维评价\n独立参考与审计')]
 for x,y,t in blocks:ax.add_patch(FancyBboxPatch((x,y),1.65,.85,boxstyle='round,pad=.03',fc='#edf4f6',ec='#8196a4'));ax.text(x+.825,y+.425,t,ha='center',va='center',fontsize=9)
 for a,b in [((1.9,1.65),(3.1,2.4)),((1.9,1.65),(3.1,.7)),((4.85,2.4),(6.15,1.65)),((4.85,.7),(6.15,1.65)),((7.9,1.65),(8.1,1.65))]:ax.annotate('',b,a,arrowprops={'arrowstyle':'->','color':'#187b91'})
 ax.text(5,3.18,'匹配数据暴露和主模型更新数；不声称匹配 FLOPs 或总优化器调用',ha='center',fontsize=11);save(fig,'01_protocol.png')
 fig,axs=plt.subplots(1,3,figsize=(10,3.4));cloud(axs[0],data['reference'],'独立真实参考')
 for ax,fam in zip(axs[1:],['vae','gan']):cloud(ax,np.load(R/f'outputs/{fam}_seed11_samples.npz')['samples'],fam.upper()+' 种子11 全部样本')
 fig.tight_layout();save(fig,'02_samples.png')
 fig,ax=plt.subplots(figsize=(8,4));
 for fam in ['vae','gan']:
  rows=[x for x in r['runs'] if x['family']==fam];ax.scatter([x['metrics']['coverage'] for x in rows],[x['metrics']['valid_fraction'] for x in rows],label=fam.upper()+' 三种子',s=80,c=C[fam],marker='o' if fam=='vae' else '^')
 for key,lab,marker in [('oracle','独立真实抽样','s'),('copier','训练集复制控制','x'),('one_mode','单模式控制','D'),('rotated_modes','旋转模式控制','P')]:
  x=r['controls'][key]['metrics'];ax.scatter(x['coverage'],x['valid_fraction'],s=100,label=lab,marker=marker)
 ax.set(xlim=(-.3,8.6),ylim=(-.05,1.08),xlabel='覆盖模式数 共8个',ylabel='有效比例 半径≤0.35',xticks=range(9));ax.legend(loc='center left',bbox_to_anchor=(1,.5),frameon=False,fontsize=9);ax.grid(alpha=.15);save(fig,'03_quality_coverage.png')
 fig,axs=plt.subplots(1,3,figsize=(11,3.4));cloud(axs[0],data['reference'],'真实八团')
 rot=np.load(R/'outputs/rotated_modes_control.npz')['samples'];cloud(axs[1],rot,'旋转22.5° 无有效样本')
 keys=['xy','radius','radial_angle'];vals=[r['controls']['rotated_modes']['feature_distance'][k] for k in keys];axs[2].bar(range(3),vals,color=['#187b91','#3c8d56','#d17b34']);axs[2].set(xticks=range(3),xticklabels=['坐标','半径','坐标+角度'],ylabel='拟合高斯特征距离',title='同一对分布 换特征');fig.tight_layout();save(fig,'04_feature_blindness.png')
 fig,ax=plt.subplots(figsize=(8,3.5));rows=r['sample_size_audit'];ax.errorbar([x['n'] for x in rows],[x['mean'] for x in rows],yerr=[x['std'] for x in rows],capsize=4,marker='o',color='#187b91');ax.set(xscale='log',xlabel='每组样本数 对数刻度',ylabel='坐标高斯距离 均值±标准差',title='同一真实分布的两次独立抽样 20次重复');ax.grid(alpha=.2);fig.tight_layout();save(fig,'05_sample_size.png')
 fig,axs=plt.subplots(1,3,figsize=(11,3.5))
 for ax,key,title in zip(axs,['vae_seed11_samples','gan_seed11_samples','copier_control'],['VAE 种子11','GAN 种子11','训练集复制控制']):
  f=np.load(R/f'outputs/{key}.npz');ax.scatter(f['train_distance'],f['holdout_distance'],s=2,alpha=.25);m=max(.05,np.quantile(np.r_[f['train_distance'],f['holdout_distance']],.98));ax.plot([0,m],[0,m],c='#d17b34',ls='--');ax.set(xlim=(-.002,m),ylim=(-.002,m),xlabel='到训练集最近距离',ylabel='到独立留出集最近距离',title=title)
 fig.tight_layout();save(fig,'06_nearest_neighbor.png')
if __name__=='__main__':main()
