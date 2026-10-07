from pathlib import Path
import os,json
R=Path(__file__).resolve().parent;os.environ.setdefault('MPLCONFIGDIR',str(R/'tmp/mpl'))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10})
F=R/'figures';F.mkdir(exist_ok=True);d=np.load(R/'outputs/data.npz');p=np.load(R/'outputs/plot_data.npz')
def save(n):plt.savefig(F/n,dpi=170,bbox_inches='tight');plt.close()
fig,axes=plt.subplots(1,4,figsize=(10,2.8));axes[0].scatter(*d['xy_test'].T,s=6);axes[0].set_title('真实二维测试样本')
for ax,c in zip(axes[1:],['standard','high_beta','restore_beta']):
 x=p[f'xy_{c}_6001_samples'];ax.scatter(*x.T,s=8);ax.set_title(c)
for ax in axes:ax.set(xlim=(-3,3),ylim=(-3,3));ax.set_aspect('equal')
save('01_xy_samples.png')
fig,axes=plt.subplots(4,8,figsize=(10,4))
for row,c in enumerate(['input','standard','high_beta','restore_beta']):
 x=d['image_test'] if row==0 else 1/(1+np.exp(-p[f'image_{c}_6001_mean_decoding']))
 for i,ax in enumerate(axes[row]):ax.imshow(x[i].reshape(8,8),vmin=0,vmax=1,cmap='gray');ax.set_xticks([]);ax.set_yticks([])
 axes[row,0].set_ylabel(c,fontsize=8)
save('02_image_reconstruction.png')
fig,axes=plt.subplots(1,2,figsize=(9,2.2))
for ax,kind in zip(axes,['xy','image']):
 for c in ['standard','high_beta','restore_beta']:
  t=p[f'{kind}_{c}_6001_trace'];ax.plot(t[:,0],t[:,2],label=c)
 ax.axvline(300,c='gray',ls='--');ax.set(title=kind,xlabel='更新步',ylabel='每例KL / nat');ax.legend(fontsize=7)
save('03_kl_training.png')
fig,axes=plt.subplots(1,3,figsize=(9,2.2))
for ax,c in zip(axes,['standard','high_beta','restore_beta']):
 z=p[f'xy_{c}_6001_mu'];ax.scatter(*z.T,s=7);ax.set(title=c,xlim=(-2,2),ylim=(-2,2),xlabel='mu1',ylabel='mu2')
save('04_posterior_means.png')
fig,ax=plt.subplots(figsize=(9,2.2));ax.axis('off')
for x,y,t in [(.1,.5,'输入 x'),(.35,.75,'编码器均值 mu'),(.35,.25,'编码器 logvar'),(.60,.5,'z = mu + sigma × epsilon'),(.88,.5,'解码器\n观测分布')]:ax.text(x,y,t,ha='center',va='center',fontsize=9,bbox=dict(boxstyle='round',fc='#e8f1f6',ec='#31617d'),transform=ax.transAxes)
for a,b in [((.16,.55),(.24,.72)),((.16,.45),(.24,.28)),((.45,.75),(.56,.58)),((.45,.25),(.56,.42)),((.75,.5),(.81,.5))]:ax.annotate('',xy=b,xytext=a,xycoords='axes fraction',arrowprops={'arrowstyle':'->'})
ax.text(.61,.07,'epsilon ~ N(0,I)，随机性在参数之外',ha='center',transform=ax.transAxes);save('05_reparameterization.png')
fig,axes=plt.subplots(2,8,figsize=(10,2.2))
for row,k in enumerate(['sample_means','samples']):
 x=p[f'image_standard_6001_{k}']
 for i,ax in enumerate(axes[row]):ax.imshow(x[i].reshape(8,8),vmin=0,vmax=1,cmap='gray');ax.axis('off')
fig.subplots_adjust(left=.12,right=.99,bottom=.05,top=.95,hspace=.22,wspace=.15)
fig.text(.015,.72,'概率均值',fontsize=14,ha='left',va='center')
fig.text(.015,.27,'Bernoulli\n样本',fontsize=14,ha='left',va='center')
save('06_image_samples.png')
