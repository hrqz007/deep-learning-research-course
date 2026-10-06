"""Regenerate only figures from recorded results; no training or metric changes."""
from pathlib import Path
import argparse,json,os
ROOT=Path(__file__).resolve().parent
for key,folder in [("MPLCONFIGDIR","mpl"),("XDG_CACHE_HOME","cache")]:
    path=ROOT/"tmp"/folder;path.mkdir(parents=True,exist_ok=True);os.environ.setdefault(key,str(path))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
FONT=Path(os.environ.get('DL_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
if not FONT.is_file():raise RuntimeError('Install Noto Sans CJK or set DL_CJK_FONT to a readable CJK font')
fp=FontProperties(fname=str(FONT));plt.rcParams.update({'font.family':fp.get_name(),'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'figure.dpi':160})
from matplotlib import font_manager
font_manager.fontManager.addfont(str(FONT))
COLORS=['#2873a3','#de8737','#4b9573','#9c67a6','#d65559']
def save(fig,out,name):
    fig.tight_layout();fig.savefig(out/name,dpi=170,bbox_inches='tight');plt.close(fig)
def main(results,output):
    source=Path(results);r=json.loads(source.read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    data=np.load(source.parent/'plot_data.npz',allow_pickle=False)
    fig,ax=plt.subplots(figsize=(10,2.8));ax.set_xlim(0,10);ax.set_ylim(0,3);ax.axis('off')
    boxes=[(.85,'观测 x\nB × 8'),(3.45,'编码器\n8 → 16 → 2'),(5.3,'瓶颈 z\nB × 2'),(7.2,'解码器\n2 → 16 → 8'),(9.2,'重构预测\nB × 8')]
    for px,label in boxes:
        ax.text(px,1.75,label,ha='center',va='center',fontsize=11,bbox=dict(boxstyle='round,pad=.5',facecolor='#eaf1f7',edgecolor='#2873a3'))
    for a,b in [(1.55,2.45),(4.2,4.8),(5.85,6.45),(8.0,8.65)]:
        ax.annotate('',xy=(b,1.75),xytext=(a,1.75),arrowprops=dict(arrowstyle='->',color='#365468',lw=1.8))
    ax.text(5,.57,'AE：输入 x，目标 x     DAE：输入 x + 噪声，目标仍为 x',ha='center',fontsize=12,color='#365468')
    ax.text(5,.05,'所有输出必须经过瓶颈；标签 y 只用于另行训练的线性探测',ha='center',fontsize=11,color='#555555')
    save(fig,out,'06_encoder_decoder.png')

    fig,axs=plt.subplots(1,3,figsize=(10,3.1));labels=data['labels']
    for ax,key,title in zip(axs,['pca2_0','ae_5801','dae_5801'],['PCA2','基础AE 5801','去噪AE 5801']):
        z=data['latent_'+key];ax.scatter(z[:,0],z[:,1],c=labels,cmap='coolwarm',s=12,alpha=.65,vmin=-1,vmax=1);ax.set(xlabel='潜在坐标1',ylabel='潜在坐标2',title=title)
    fig.suptitle('同一320测试样本；颜色为真实标签，仅在评价时使用',y=1.06);save(fig,out,'01_latent.png')
    fig,ax=plt.subplots(figsize=(7.5,3.1));e=r['pca_train_eigenvalues'];ax.bar(np.arange(1,9),e,color=COLORS[0]);ax.set(xticks=np.arange(1,9),xlabel='主成分次序',ylabel='训练协方差特征值',title='均值中心化，未按坐标标准化');save(fig,out,'02_spectrum.png')
    fig,ax=plt.subplots(figsize=(8,3.1))
    for row in r['results']:
        if row['representation'] in ['ae','dae']:
            ax.plot([t['step'] for t in row['trace']],[t['clean_validation_mse'] for t in row['trace']],label=f"{row['representation']} {row['seed']}",linestyle='-' if row['representation']=='ae' else '--',alpha=.8)
    ax.set(xlabel='训练步',ylabel='干净验证重构MSE',yscale='log');ax.legend(ncol=3,fontsize=9);save(fig,out,'03_training.png')
    names=['raw','pca2','random2','ae','dae'];fig,axs=plt.subplots(1,2,figsize=(9,3.4))
    for i,name in enumerate(names):
        rows=[x for x in r['results'] if x['representation']==name]
        for ax,key in zip(axs,['test_reconstruction_mse','probe_accuracy']):
            vals=[x[key] for x in rows];ax.bar(i,np.mean(vals),color=COLORS[i],alpha=.65);ax.scatter(np.linspace(i-.12,i+.12,len(vals)),vals,color='#18202a',s=20,zorder=3)
    for ax in axs:ax.set(xticks=range(5),xticklabels=names)
    axs[0].set(ylabel='干净测试重构MSE');axs[1].set(ylabel='固定线性探测测试准确率',ylim=(0,1.08));axs[1].axhline(.5,color='gray',ls=':',label='平衡标签0.5基线');axs[1].legend(fontsize=8);save(fig,out,'04_reconstruction_probe.png')
    fig,ax=plt.subplots(figsize=(8,3.1));names=['raw','pca2','ae','dae']
    for i,name in enumerate(names):
        vals=[x['corrupted_test_denoising_mse'] for x in r['results'] if x['representation']==name];ax.bar(i,np.mean(vals),color=COLORS[i],alpha=.65);ax.scatter(np.linspace(i-.12,i+.12,len(vals)),vals,color='#18202a',s=22)
    ax.set(xticks=range(4),xticklabels=names,ylabel='带噪输入 对干净目标MSE',title='固定额外Gaussian噪声 标准差0.5');save(fig,out,'05_denoising.png')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',default='outputs/results.json');p.add_argument('--output',default='figures');a=p.parse_args();main(a.results,a.output)
