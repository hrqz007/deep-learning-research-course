"""Original explanatory figures for the fixed unit-023 experiment.

All fixed inputs/results are checked and every PNG is rendered in memory before
any output file is replaced. Requires matplotlib and a CJK font.
"""
from pathlib import Path
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import experiment as e

FONT=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.unicode_minus':False,'figure.facecolor':'white','savefig.facecolor':'white'})
BLUE='#2563a6';ORANGE='#d07822';GREEN='#148a73';RED='#c64a55';PURPLE='#7653aa'


def build(config=e.BASE/'data/config.json', samples=e.BASE/'data/samples.csv',
          results=e.BASE/'outputs', output=e.BASE/'figures'):
    summary,trace,predictions,checks=e.teaching_payload(config,samples,results)
    rows=e.load_samples(samples);X,y=e.split_arrays(rows,'train');payload={}
    def save(fig,name):
        b=io.BytesIO();fig.savefig(b,format='png',dpi=170,bbox_inches='tight',metadata={'Software':'Unit 023 original figures'})
        payload[name]=b.getvalue();plt.close(fig)
    fig,ax=plt.subplots(figsize=(10.6,3.7));ax.set(xlim=(0,10.6),ylim=(0,3.7));ax.axis('off')
    boxes=[(.15,2.35,1.9,.75,'固定训练数据\nX (n,d), y (n,)',BLUE),
           (2.65,2.35,1.8,.75,'预测\nA @ theta: (n,)',BLUE),
           (5.05,2.35,1.8,.75,'残差\nr = pred - y',ORANGE),
           (7.65,2.35,2.2,.75,'目标\nL = r @ r / (2n)',ORANGE),
           (5.0,.7,2.2,.85,'梯度\ng = A.T @ r / n',PURPLE),
           (2.35,.7,2.1,.85,'同步更新\ntheta <- theta - eta*g',GREEN)]
    for xx,yy,w,h,t,c in boxes:
        ax.add_patch(FancyBboxPatch((xx,yy),w,h,boxstyle='round,pad=.07',facecolor=c+'12',edgecolor=c,lw=1.6));ax.text(xx+w/2,yy+h/2,t,ha='center',va='center',fontsize=10)
    for p,q in [((2.1,2.72),(2.58,2.72)),((4.52,2.72),(4.98,2.72)),((6.92,2.72),(7.58,2.72)),((8.75,2.28),(7.0,1.3)),((4.93,1.13),(4.52,1.13)),((3.4,1.62),(3.4,2.28))]:
        ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=14,color='#546579'))
    ax.text(8.5,.75,'验证与测试：只评价\n没有箭头回到本讲更新',color=RED,ha='center',fontsize=10)
    ax.text(5.3,3.45,'一轮更新沿训练闭环运行；记录参数 t 与目标 t 对齐',ha='center',weight='bold')
    save(fig,'01_loop.png')

    fig,axs=plt.subplots(1,2,figsize=(10.6,3.7));xx=np.linspace(-1.25,1.25,200)
    axs[0].scatter([-1,0,1],[-1,1,3],c='black',label='三条标签',zorder=4)
    for w,b,label,col in [(0,0,'初始 w=0, b=0',RED),(2/3,.5,'一步 w=2/3, b=1/2',BLUE),(2,1,'最优 w=2, b=1',GREEN)]:axs[0].plot(xx,w*xx+b,label=label,color=col)
    axs[0].set(xlabel='x',ylabel='y 与预测',title='手算更新真的改变预测');axs[0].legend(fontsize=9)
    axs[1].bar(np.arange(3)-.18,[1,-1,-3],.36,color=RED,label='更新前残差')
    axs[1].bar(np.arange(3)+.18,[5/6,-1/2,-11/6],.36,color=BLUE,label='更新后残差')
    axs[1].axhline(0,color='gray',lw=.8);axs[1].set(xticks=range(3),xticklabels=['x=-1','x=0','x=1'],ylabel='预测 - 标签',title='L: 11/6 → 155/216');axs[1].legend(fontsize=9)
    fig.tight_layout();save(fig,'02_hand_update.png')

    fig,axs=plt.subplots(1,2,figsize=(10.6,3.6));contrib=np.array([[-1/3,1/3],[0,-1/3],[-1,-1]])
    im=axs[0].imshow(contrib,cmap='RdBu',vmin=-1,vmax=1,aspect='auto')
    for i in range(3):
        for j in range(2):axs[0].text(j,i,f'{contrib[i,j]:.3f}',ha='center',va='center',color='white' if abs(contrib[i,j])>.5 else 'black')
    axs[0].set(xticks=[0,1],xticklabels=['x_i r_i / n','r_i / n'],yticks=[0,1,2],yticklabels=['样本 1','样本 2','样本 3'],title='每条路径对两个参数的贡献')
    axs[1].bar(['w 的梯度','b 的梯度'],contrib.sum(axis=0),color=[PURPLE,ORANGE]);axs[1].axhline(0,color='gray',lw=.8)
    axs[1].set(ylim=(-1.55,.15),ylabel='沿样本轴相加',title='同一个参数被所有样本共享')
    axs[1].text(0,-1.4,'-4/3',ha='center',color=PURPLE);axs[1].text(1,-1.1,'-1',ha='center',color=ORANGE)
    fig.tight_layout();save(fig,'03_gradient_paths.png')

    fig,axs=plt.subplots(1,2,figsize=(10.6,3.8));wrong=np.array([-1,1,3])[:,None]-np.array([-1,1,3])
    axs[0].imshow(np.zeros((3,1)),cmap='RdBu',vmin=-4,vmax=4,aspect='auto')
    for i in range(3):axs[0].text(0,i,'0',ha='center',va='center')
    axs[0].set(xticks=[0],xticklabels=['每行对应标签'],yticks=[0,1,2],yticklabels=['预测 -1','预测 1','预测 3'],title='正确 (3,) - (3,) → (3,)')
    axs[1].imshow(wrong,cmap='RdBu',vmin=-4,vmax=4)
    for i in range(3):
        for j in range(3):axs[1].text(j,i,str(wrong[i,j]),ha='center',va='center',color='white' if abs(wrong[i,j])>2 else 'black')
    axs[1].set(xticks=range(3),xticklabels=['标签 -1','标签 1','标签 3'],yticks=range(3),yticklabels=['预测 -1','预测 1','预测 3'],title='错误 (3,1) - (3,) → (3,3)')
    fig.suptitle('预测完全正确仍会得到错误 MSE = 16/3',fontsize=12,color=RED);fig.tight_layout();save(fig,'04_shape_bug.png')

    fig,axs=plt.subplots(1,2,figsize=(10.6,3.7))
    for eta,col in [(.5,BLUE),(2.,ORANGE),(2.1,RED)]:
        _,tr=e.train([[0]],[1],eta,22,[0,0]);axs[0].plot([r['theta'][1] for r in tr],'.-',color=col,label=f'eta={eta:g}');axs[1].semilogy([r['loss'] for r in tr],'.-',color=col,label=f'eta={eta:g}')
    axs[0].axhline(1,color=GREEN,ls='--',label='最优 b=1');axs[0].set(xlabel='步数 t',ylabel='b_t',title='同一目标，不同更新步长')
    axs[1].set(xlabel='步数 t',ylabel='L = (b-1)^2 / 2',title='边界步长并不收敛');axs[0].legend(fontsize=9);axs[1].legend(fontsize=9)
    fig.tight_layout();save(fig,'05_steps.png')

    fig,axs=plt.subplots(1,2,figsize=(10.6,3.7));ts=[r['step'] for r in trace]
    axs[0].semilogy(ts,[r['loss'] for r in trace],color=BLUE,label='全批次训练目标')
    axs[0].axhline(summary['reference']['loss'],color=GREEN,ls='--',label='lstsq 的同一训练目标');axs[0].set(xlabel='步数 t',ylabel='半 MSE',title='非零平台可以是正确答案');axs[0].legend(fontsize=9)
    axs[1].semilogy(ts,[r['gradient_norm'] for r in trace],color=PURPLE);axs[1].set(xlabel='步数 t',ylabel='梯度二范数',title='补充检查：仍有多大更新驱动力')
    fig.tight_layout();save(fig,'06_convergence.png')

    fig,axs=plt.subplots(1,2,figsize=(10.6,3.7));theta=np.array(summary['theta'])
    for z,col in [(-1,BLUE),(1,ORANGE)]:
        mask=X[:,1]==z;axs[0].scatter(X[mask,0],y[mask],color=col,label=f'标签 x2={z}')
        xx=np.linspace(-2.2,3.2,100);axs[0].plot(xx,theta[0]*xx+theta[1]*z+theta[2],color=col,label=f'模型 x2={z}')
    axs[0].axhline(summary['baseline'],color=PURPLE,ls='--',label='训练均值常数 2.5');axs[0].set(xlabel='x1',ylabel='y 与预测',title='直线利用输入，常数忽略输入');axs[0].legend(fontsize=8,ncol=2)
    pos=np.arange(3)
    for offset,key,col in [(-.18,'constant',PURPLE),(.18,'trained',GREEN)]:axs[1].bar(pos+offset,[summary['mse'][s][key] for s in e.SPLITS],.36,color=col,label=key)
    axs[1].set(xticks=pos,xticklabels=['训练','验证','测试'],yscale='log',ylabel='MSE（对数坐标）',title='同一拆分，同一评价尺度');axs[1].legend(fontsize=9)
    fig.tight_layout();save(fig,'07_baseline.png')

    fig,axs=plt.subplots(1,2,figsize=(10.6,3.7));res=e.predict(X,np.array(summary['reference_theta']))-y
    grid=res.reshape(6,2).T
    axs[0].imshow(grid,cmap='RdBu',vmin=-.5,vmax=.5,aspect='auto')
    for i in range(2):
        for j in range(6):axs[0].text(j,i,f'{grid[i,j]:.1f}',ha='center',va='center',color='white' if abs(grid[i,j])>.3 else 'black')
    axs[0].set(xticks=range(6),xticklabels=['-2','-1','0','1','2','3'],yticks=[0,1],yticklabels=['x2=-1','x2=1'],xlabel='x1',title='残差是有结构的交互项')
    sums=e.design(X).T@res
    axs[1].bar(['sum x1*r','sum x2*r','sum r'],sums,color=[BLUE,ORANGE,GREEN]);axs[1].axhline(0,color='gray',lw=.8);axs[1].set(ylim=(-1e-12,1e-12),ylabel='数值内积（约为零）',title='非零残差仍可与每个设计列正交')
    fig.tight_layout();save(fig,'08_residual_structure.png')

    fig,axs=plt.subplots(1,2,figsize=(10.6,3.7));xx=np.linspace(-1,5,100)
    axs[0].plot(xx,2-xx,color=BLUE,label='w1+w2=2，b=1');axs[0].scatter([1,4],[1,-2],c=[GREEN,RED],s=60,zorder=3)
    axs[0].annotate('lstsq: (1,1)',(1,1),xytext=(.1,2.5),arrowprops={'arrowstyle':'->'},fontsize=9)
    axs[0].annotate('初始差值 6 保留: (4,-2)',(4,-2),xytext=(1,-3.2),arrowprops={'arrowstyle':'->'},fontsize=9)
    axs[0].set(xlabel='w1',ylabel='w2',title='重复特征：一整条线都给相同预测');axs[0].set_aspect('equal',adjustable='box')
    small=1e-8;x=np.array([-1.,0.,1.]);q=np.array([1.,-2.,1.]);near=np.column_stack((x,x+small*q));A=e.design(near)
    y0=A@np.array([2.,-1.,.5]);y1=y0+small*q
    t0,_=e.least_squares(near,y0);t1,_=e.least_squares(near,y1)
    axs[1].bar(np.arange(3)-.18,t0,.36,color=BLUE,label='原标签');axs[1].bar(np.arange(3)+.18,t1,.36,color=ORANGE,label='标签仅改变 ±1e-8 / -2e-8')
    axs[1].set(xticks=range(3),xticklabels=['w1','w2','b'],ylabel='参数值',title='近重复特征：数据本身对参数不敏感');axs[1].legend(fontsize=8)
    fig.tight_layout();save(fig,'09_identifiability.png')
    output=e.output_preflight(output,payload,[config,samples,Path(__file__)])
    # All computations and PNG encodings completed before first output mutation.
    output.mkdir(parents=True,exist_ok=True)
    for name,value in payload.items():(output/name).write_bytes(value)
    return list(payload)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=e.BASE/'data/config.json')
    parser.add_argument('--samples',type=Path,default=e.BASE/'data/samples.csv')
    parser.add_argument('--results',type=Path,default=e.BASE/'outputs')
    parser.add_argument('--output',type=Path,default=e.BASE/'figures');args=parser.parse_args()
    print('\n'.join(build(args.config,args.samples,args.results,args.output)))
