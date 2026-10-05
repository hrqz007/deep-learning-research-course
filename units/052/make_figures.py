"""Regenerate original figures with all legends outside the data axes."""
from pathlib import Path
import argparse,json,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from experiment import ROOT
FONT=os.environ.get('DL_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if not Path(FONT).is_file():raise RuntimeError('Install Noto Sans CJK or set DL_CJK_FONT to an existing font file')
plt.rcParams.update({'font.family':[FontProperties(fname=FONT).get_name(),'DejaVu Sans'],'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
BLUE='#2166ac';RED='#b44932';GREEN='#278565';GRAY='#6b7280'
def heat(ax,z,title,fmt='.3f',vmin=None,vmax=None,cmap='Blues'):
    z=np.array(z);ax.imshow(z,cmap=cmap,vmin=vmin,vmax=vmax,aspect='auto');ax.set_title(title);ax.set_xticks(range(z.shape[1]));ax.set_yticks(range(z.shape[0]))
    for i,j in np.ndindex(z.shape):ax.text(j,i,format(z[i,j],fmt),ha='center',va='center',fontsize=10,color='black',bbox={'facecolor':'white','alpha':.8,'edgecolor':'none','pad':1})
def legend(fig,ax,n=2):fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=n,frameon=False);fig.tight_layout(rect=[0,0,1,.84])
def flow(r):
    fig,ax=plt.subplots(figsize=(11,4.2));ax.axis('off');ax.set(xlim=(0,11),ylim=(0,4.2))
    nodes={'X':(.3,1.5,'X [B,L,D]\n每行一个位置'),'Q':(2.65,2.75,'Q=XWQ\n[B,L,dk]'),'K':(2.65,1.5,'K=XWK\n[B,S,dk]'),'V':(2.65,.25,'V=XWV\n[B,S,dv]'),'S':(5.25,2.2,'S=QK^T/√dk\n[B,L,S]'),'A':(7.75,2.2,'A=softmax(S+M)\n沿key轴归一化'),'O':(7.75,.25,'O=AV\n[B,L,dv]')}
    for x,y,t in nodes.values():ax.add_patch(FancyBboxPatch((x,y),2.05,.87,boxstyle='round,pad=.06',fc='#eef4f8',ec=BLUE));ax.text(x+1.025,y+.435,t,ha='center',va='center')
    for a,b in [('X','Q'),('X','K'),('X','V'),('Q','S'),('K','S'),('S','A'),('V','O')]:
        x,y,_=nodes[a];xx,yy,_=nodes[b];ax.add_patch(FancyArrowPatch((x+2.1,y+.43),(xx-.06,yy+.43),arrowstyle='->',mutation_scale=13,color=GRAY))
    ax.add_patch(FancyArrowPatch((8.77,2.15),(8.77,1.2),arrowstyle='->',mutation_scale=13,color=GRAY));ax.text(5.6,1.38,'决定从哪里取\n再组合实际内容',color=RED,ha='center');fig.tight_layout();return fig
def matrices(r):
    s=r['hand'][0];fig,axs=plt.subplots(2,3,figsize=(10,5))
    for b in range(2):
        for j,key in enumerate(['score','a','prediction']):heat(axs[b,j],s[key][b],f'样本{b+1} '+{'score':'分数S','a':'权重A','prediction':'输出O'}[key]);axs[b,j].set_ylabel('query位置');axs[b,j].set_xlabel('key位置' if j<2 else '值坐标')
    fig.tight_layout();return fig
def jacobian(r):
    fig,axs=plt.subplots(1,2,figsize=(9,3.7));heat(axs[0],r['hand'][0]['jacobian'][0][0],'样本1位置1 完整Jacobian',vmin=-.25,vmax=.25,cmap='RdBu');axs[0].set(xlabel='被改变的分数',ylabel='受到影响的权重');z=np.linspace(-5,5,201);p=1/(1+np.exp(-z));axs[1].plot(z,p*(1-p),label='对角 p(1-p)',color=BLUE);axs[1].plot(z,-p*(1-p),label='非对角 -p(1-p)',color=RED);axs[1].set(xlabel='两分数之差',ylabel='局部导数',title='竞争使另一个权重反向变化');legend(fig,axs[1]);return fig
def gradient(r):
    s=r['hand'][0];paths=np.array(s['input_gradient_paths']);fig,axs=plt.subplots(1,2,figsize=(10,3.8));x=np.arange(4)
    for j,(lab,col) in enumerate(zip(['Q路径','K路径','V路径'],[BLUE,RED,GREEN])):axs[0].bar(x+(j-1)*.23,paths[j,:,:,0].ravel(),.22,label=lab,color=col)
    axs[0].set_xticks(x,['样1位1','样1位2','样2位1','样2位2']);axs[0].set(ylabel='输入特征1的梯度',title='同一输入同时走三条路径');p=np.array(s['weight_gradient_per_sample']).squeeze(-1)
    for b,col in enumerate([BLUE,RED]):axs[1].bar(np.arange(6)+(b-.5)*.33,p[:,b,:].ravel(),.32,label=f'样本{b+1}贡献',color=col)
    axs[1].set_xticks(range(6),['Q1','Q2','K1','K2','V1','V2']);axs[1].set(ylabel='参数梯度贡献',title='位置求和后跨样本相加');h,l=axs[0].get_legend_handles_labels();hh,ll=axs[1].get_legend_handles_labels();fig.legend(h+hh,l+ll,loc='upper center',ncol=5,frameon=False,fontsize=10);fig.tight_layout(rect=[0,0,1,.84]);return fig
def masks(r):
    causal=np.tril(np.ones((4,4),dtype=int));key=np.ones((4,4),dtype=int);key[:,3]=0;both=causal*key;used=both.copy();used[3]=0;fig,axs=plt.subplots(1,4,figsize=(11,3.5))
    for ax,z,title in zip(axs,[causal,key,both,used],['因果允许 j≤i','key有效 前3列','两约束取交集','无效query行约定为零']):heat(ax,z,title,'d',0,1);ax.set(xlabel='key位置',ylabel='query位置')
    fig.tight_layout();return fig
def leakage(r):
    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    for ax,name in zip(axs,['causal','unmasked']):
        z=r['leakage'][name];ax.plot(range(4),np.array(z['before'])[0,:,0],'o-',label='修改未来前',color=BLUE);ax.plot(range(4),np.array(z['after'])[0,:,0],'s--',label='修改位置2和3后',color=RED);ax.axvline(1.5,color=GRAY,ls=':');ax.set(xlabel='query位置（0起）',ylabel='输出特征1',title=f'{name} 前缀最大差={z["prefix_max_change"]:.4f}',xticks=range(4))
    legend(fig,axs[0]);return fig
def experiment(r):
    fig,ax=plt.subplots(figsize=(10,4.4));labels=[];values=[]
    for alpha in [0,8]:
        for cond in ['causal','unmasked']:
            group=[a for a in r['runs'] if a['alpha']==alpha and a['condition']==cond];values.append([a['test_mse'] for a in group]);labels.append(f'α={alpha}\n'+('因果合法' if cond=='causal' else '无mask 泄漏'))
    mean=np.mean(values,axis=1);ax.bar(range(4),mean,color=[BLUE,RED,BLUE,RED],width=.65,label='3种子均值')
    for i,v in enumerate(values):ax.scatter([i-.13,i,i+.13],v,color='black',s=18,zorder=3)
    base=np.mean([z['zero_baseline_test_mse'] for z in r['runs'][::4]]);ax.axhline(base,color=GREEN,ls='--',label='零基线均值');ax.set_xticks(range(4),labels);ax.set(yscale='log',ylabel='测试MSE 对数轴',title='新测试序列仍可能暴露未来答案',ylim=(1e-7,4))
    for i,v in enumerate(mean):ax.text(i,v*1.35,f'{v:.3g}',ha='center')
    legend(fig,ax);return fig
def scaling(r):
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    for scaled,color in [(False,RED),(True,BLUE)]:
        a=[x for x in r['scaling'] if x['scaled']==scaled]
        for ax,key in zip(axs,['score_variance','mean_entropy_nats']):ax.plot([v['d'] for v in a],[v[key] for v in a],'o-',label='除以√dk' if scaled else '不缩放',color=color);ax.set(xscale='log',xlabel='dk（对数轴）',xticks=[1,4,16,64]);ax.set_xticklabels([1,4,16,64])
    axs[0].set(ylabel='经验分数方差',title='独立零均值单位方差条件');axs[1].set(ylabel='平均行熵 nats',title='每行16个key');axs[1].axhline(np.log(16),color=GRAY,ls=':',lw=1);legend(fig,axs[0]);return fig
def nonunique(r):
    fig,axs=plt.subplots(1,2,figsize=(9,3.5));x=np.arange(3);a=np.array([.4,.2,.4]);b=np.array([.2,.6,.2]);v=np.array([0.,1.,2.]);axs[0].bar(x-.16,a,.31,label='权重A',color=BLUE);axs[0].bar(x+.16,b,.31,label='权重B',color=RED);axs[0].set(xticks=x,xlabel='key位置',ylabel='权重',title='两组严格正权重');axs[1].bar([0,1],[a@v,b@v],color=[BLUE,RED]);axs[1].set(xticks=[0,1],xticklabels=['A·V','B·V'],ylim=(0,1.2),ylabel='相同输出',title='V=(0,1,2) 都输出1');legend(fig,axs[0]);return fig
BUILDERS={'01_flow':flow,'02_matrices':matrices,'03_jacobian':jacobian,'04_gradient':gradient,'05_masks':masks,'06_leakage':leakage,'07_experiment':experiment,'08_scaling':scaling,'09_nonunique':nonunique}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'figures');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);r=json.loads((ROOT/'outputs/results.json').read_text())
    for name,fn in BUILDERS.items():fig=fn(r);fig.savefig(args.output/(name+'.png'),dpi=170,bbox_inches='tight');plt.close(fig);print(name)
