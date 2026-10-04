"""Rebuild eight original explanatory figures from the verified default experiment."""
from pathlib import Path
import csv
import hashlib
import json
HERE=Path(__file__).resolve().parent
EXPECTED = {'data/batch.csv': '42807a0698a93257e6eb239ede3d1fca4c25487fabb21df777194de74ab1074d', 'data/config.json': '949093a235057e2a6ccc725dd703814d93c5f297bd50f5dee139f13ce2e3107f', 'outputs/fd_sweep.csv': '8c0b267ef8074f210fcb44d78512b43a877919bd9154f680854404852a987443', 'outputs/gradients.csv': '3b67ab76187a899999ce41b66c2f585806f7a68c629ad43bd1a960ab8ecce406', 'outputs/sample_gradients.csv': 'db1459be7b12a6b772d080ff848ff9fc423dc02126cd97c6875839d870fda3bf', 'outputs/summary.json': '5f2a62d010169df1a687fb6fadfe3239ba3a1f5456da44be5ea0dac1aeec21b1', 'outputs/tensors.json': '3ba337cf99e983816775744560613842d5fe17f022200d7dc8ac409d2399d811'}

def verify_default():
    for relative,digest in EXPECTED.items():
        path=HERE/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'default fixture changed: {relative}; use custom outputs without fixed figures')

def main():
    verify_default()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.font_manager import fontManager, FontProperties
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if not font.is_file():
        raise RuntimeError('Install Noto Sans CJK before rebuilding Chinese figures')
    fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':FontProperties(fname=str(font)).get_name(),'axes.unicode_minus':False,'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    from experiment import load_inputs,loss_and_grad
    ids,X,y,params,cfg=load_inputs();r=loss_and_grad(X,y,params)
    figdir=HERE/'figures';figdir.mkdir(exist_ok=True)
    blue='#176caa';red='#c14f45';green='#168273';gray='#617386'
    def save(fig,name):
        fig.savefig(figdir/name,dpi=180,bbox_inches='tight',pad_inches=.15);plt.close(fig)
    def box(ax,x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02',fc=color,ec='none'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11,color='#172636')
    fig,ax=plt.subplots(figsize=(10.8,3.4));ax.axis('off');ax.set(xlim=(0,11),ylim=(0,3.4))
    labels=['X = H0\n(5, 2)','Z1 → tanh → H1\n(5, 3)','Z2 → tanh → H2\n(5, 2)','Z3 = logits\n(5, 3)']
    for i,t in enumerate(labels):
        box(ax,i*2.8,.7,2.45,1,t,'#e2f0fa')
        if i<3:
            ax.annotate('',xy=(i*2.8+2.76,1.2),xytext=(i*2.8+2.47,1.2),arrowprops={'arrowstyle':'->','color':blue,'lw':2})
            ax.text(i*2.8+2.58,2.27,f'W{i+1}, b{i+1}\n{params[i]["W"].shape}, {params[i]["b"].shape}',ha='center',fontsize=10,color=gray)
    for i,t in enumerate(['dX (5, 2)','D1 (5, 3)','D2 (5, 2)','D3 = (P − Y) / 5']):
        ax.text(i*2.8+1.22,.22,t,ha='center',color=red,fontsize=11)
        if i:ax.annotate('',xy=(i*2.8-.25,.25),xytext=(i*2.8+.06,.25),arrowprops={'arrowstyle':'->','color':red,'lw':2})
    ax.text(0,3.1,'同一参数快照：先完成全部前向，再完成全部反向',color=gray,fontsize=14)
    save(fig,'01_shapes.png')
    fig,ax=plt.subplots(figsize=(10,3.2));ax.axis('off');ax.set(xlim=(0,10),ylim=(0,3.2))
    for i,v in enumerate(r['losses']):box(ax,.1+i*1.75,1.72,1.5,.72,f'样本 {i+1}\nloss = {v:.4f}','#e3f1fa')
    box(ax,1.4,.15,3,.78,f'mean = {r["loss"]:.6f}\n上游系数：每条 1/5','#dcf0e9')
    box(ax,5.5,.15,3,.78,f'sum = {r["loss"]*5:.6f}\n上游系数：每条 1','#fae9df')
    for i in range(5):
        for end,color in ((2.9,green),(7.,red)):ax.annotate('',xy=(end,.97),xytext=(.85+i*1.75,1.68),arrowprops={'arrowstyle':'->','lw':.8,'color':color,'alpha':.45})
    ax.text(.1,2.85,'类别轴 → 每条交叉熵（已完成）        样本轴 → 目标标量（在此选择）',fontsize=12,color=gray)
    save(fig,'02_reduction.png')
    fig,axes=plt.subplots(1,3,figsize=(10,3.2),gridspec_kw={'wspace':.45})
    arrays=[np.array([[-.75,.75],[-.75,.75]]),np.array([[-.25,.25],[-.25,.25]]),np.array([[-1.,1.],[-1.,1.]])]
    for ax,a,title in zip(axes,arrays,['样本1：h1 外积 d1','样本2：h2 外积 d2','两条路径相加：dW2']):
        ax.imshow(a,cmap='RdBu',vmin=-1,vmax=1);ax.set_title(title,fontsize=12);ax.set_xticks([0,1],['输出0','输出1']);ax.set_yticks([0,1],['隐藏0','隐藏1'])
        for ij in np.ndindex(a.shape):ax.text(ij[1],ij[0],str(a[ij]),ha='center',va='center',color='white' if abs(a[ij])>.7 else '#172636',fontsize=16)
    save(fig,'03_outer_products.png')
    fig,axes=plt.subplots(1,2,figsize=(10,3.4));z=np.linspace(-3,3,301)
    axes[0].plot(z,np.tanh(z),label='tanh(z)',color=blue,lw=2);axes[0].plot(z,1-np.tanh(z)**2,label='导数 1 − tanh²(z)',color=red,lw=2)
    axes[1].plot(z,np.maximum(z,0),label='ReLU(z)',color=blue,lw=2);axes[1].plot([-3,0],[0,0],color=red,lw=2,label='反传约定');axes[1].plot([0,3],[1,1],color=red,lw=2)
    axes[1].scatter([0],[1],facecolor='white',edgecolor=red,s=80,zorder=5);axes[1].scatter([0],[0],color='black',s=35,zorder=6)
    axes[1].annotate('z = 0 无普通导数\n程序约定选 0',xy=(0,0),xytext=(-2.8,1.8),arrowprops={'arrowstyle':'->','color':gray},fontsize=11)
    for ax in axes:ax.axhline(0,color='#aeb7bf',lw=.6);ax.axvline(0,color='#aeb7bf',lw=.6);ax.set_xlabel('激活前 z');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.27),frameon=False,fontsize=11);ax.grid(alpha=.15)
    save(fig,'04_activations.png')
    fig,axes=plt.subplots(1,3,figsize=(10,2.9),gridspec_kw={'width_ratios':[1.5,1.15,1]})
    a=np.arange(1,7).reshape(2,3);axes[0].imshow(a,cmap='Blues',vmin=0,vmax=7)
    for ij in np.ndindex(a.shape):axes[0].text(ij[1],ij[0],str(a[ij]),ha='center',va='center',fontsize=17)
    axes[0].set_title('广播后的上游 G：(2, 3)');axes[0].set_xticks(range(3),['特征0','特征1','特征2']);axes[0].set_yticks(range(2),['样本0','样本1'])
    for ax in axes[1:]:ax.axis('off')
    axes[1].text(.02,.84,'原shape (3,)\n沿样本轴求和\n→ [5, 7, 9]',fontsize=14,color=blue,va='top');axes[1].text(.02,.35,'原shape (2, 1)\n沿特征轴求和\n→ [[6], [15]]',fontsize=14,color=green,va='top')
    axes[2].text(.02,.84,'原shape ()\n全部求和\n→ 21',fontsize=14,color=red,va='top');axes[2].text(.02,.35,'原shape (2, 3)\n没有广播复制\n→ G 本身',fontsize=14,color=gray,va='top')
    save(fig,'05_broadcast.png')
    fig,ax=plt.subplots(figsize=(9.5,3.2));ind=np.arange(5)
    ax.bar(ind-.18,[.2]*5,width=.36,color=green,label='正确：按微批大小加权');ax.bar(ind+.18,[.25,.25,1/6,1/6,1/6],width=.36,color=red,label='错误：两个微批均值再平均')
    ax.axvline(1.5,color=gray,ls='--');ax.set_xticks(ind,['s1','s2','s3','s4','s5']);ax.set_ylim(0,.32);ax.set_ylabel('每条样本的有效权重');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.22),ncol=2,frameon=False,fontsize=11)
    ax.text(.5,.29,'微批A：2条',ha='center',color=gray);ax.text(3,.29,'微批B：3条',ha='center',color=gray);save(fig,'06_microbatch_weights.png')
    fig,ax=plt.subplots(figsize=(9.5,3.3));names=[];values=[]
    for j,p in enumerate(r['grads']):
        for key in ('W','b'):names.append(f'{key}{j+1}');values.append(np.linalg.norm(p[key]))
    duplicated=loss_and_grad(np.repeat(X,3,axis=0),np.repeat(y,3),params);dup=[np.linalg.norm(p[k]) for p in duplicated['grads'] for k in ('W','b')];q=np.arange(6)
    ax.bar(q-.25,values,width=.25,label='mean',color=blue);ax.bar(q,np.array(values)*5,width=.25,label='sum',color=red);ax.bar(q+.25,dup,width=.25,label='整批复制3次后的 mean',color=green)
    ax.set_xticks(q,names);ax.set_ylabel('参数块梯度的L2范数');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.2),ncol=3,frameon=False);ax.grid(axis='y',alpha=.2);save(fig,'07_gradient_scales.png')
    sweep=list(csv.DictReader((HERE/'outputs/fd_sweep.csv').open()));fd=list(csv.DictReader((HERE/'outputs/gradients.csv').open()))
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),gridspec_kw={'wspace':.35})
    axes[0].loglog([float(a['h']) for a in sweep],[float(a['max_absolute_error']) for a in sweep],marker='o',color=blue);axes[0].set_xlabel('中心差分 h（对数轴）');axes[0].set_ylabel('26坐标最大绝对误差');axes[0].grid(alpha=.25);axes[0].set_title('截断误差与舍入误差的折中',fontsize=12)
    aa=[float(a['analytic']) for a in fd];nn=[float(a['numeric']) for a in fd];lo=min(aa+nn)-.02;hi=max(aa+nn)+.02
    axes[1].plot([lo,hi],[lo,hi],color=red,lw=1.2);axes[1].scatter(aa,nn,c=green,s=32,zorder=3);axes[1].set(xlabel='解析梯度',ylabel='中心差分梯度',xlim=(lo,hi),ylim=(lo,hi));axes[1].set_title('h = 1e−5，全部26个坐标',fontsize=12);axes[1].grid(alpha=.2);save(fig,'08_finite_difference.png')
    print('8 original figures rebuilt from verified default inputs and results')
if __name__=='__main__':main()
