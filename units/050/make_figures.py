"""Original diagrams and source-backed plots. Legends occupy external bands."""
from pathlib import Path
import json,tempfile,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch
ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures'
from matplotlib import font_manager
try: font_manager.findfont('Noto Sans CJK SC',fallback_to_default=False)
except ValueError:
    from fontTools.ttLib import TTCollection
    path=Path(tempfile.gettempdir())/'dl050-noto-sc.ttf'
    if not path.exists():
        collection=TTCollection('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
        next(f for f in collection.fonts if f['name'].getDebugName(1)=='Noto Sans CJK SC').save(path)
    font_manager.fontManager.addfont(str(path))
plt.rcParams.update({'font.family':['Noto Sans CJK SC','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':160})
C={'full':'#2166ac','full_clip':'#168575','tbptt4_clip':'#c76b25'}
L={'full':'完整 BPTT','full_clip':'完整 BPTT + 裁剪','tbptt4_clip':'截断 K=4 + 裁剪'}

def save(fig,name):
    fig.savefig(FIG/name,facecolor='white');plt.close(fig)
def box(ax,xy,text,width=1.8,height=.7,color='#e8eff7'):
    x,y=xy;ax.add_patch(FancyBboxPatch((x,y),width,height,boxstyle='round,pad=0.03',fc=color,ec='#536b82'));ax.text(x+width/2,y+height/2,text,ha='center',va='center',fontsize=11)
def arrow(ax,start,end,color='#526779',style='->'):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle=style,mutation_scale=14,color=color,lw=1.5))
def legend(fig,handles,labels,ncol=3):fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.025),ncol=ncol,frameon=False,fontsize=10)

def main(output=None):
    global FIG
    FIG=Path(output) if output else ROOT/'figures'
    FIG.mkdir(parents=True,exist_ok=True)
    h=json.loads((ROOT/'outputs/hand_calculation.json').read_text());r=h['reference'];runs=json.loads((ROOT/'outputs/results.json').read_text())['runs'];d=np.load(ROOT/'data/delayed_sign.npz',allow_pickle=False)
    fig,ax=plt.subplots(figsize=(11,4.8));ax.set(xlim=(0,11),ylim=(0,4.8));ax.axis('off')
    box(ax,(.15,2.1),'h0 = 0',1.3)
    for i in range(3):
        x=2.2+3*i;box(ax,(x,2.1),f'h{i+1} = tanh(a{i+1})');box(ax,(x,.5),f'x{i+1}  输入',color='#edf5eb');box(ax,(x,3.6),f'z{i+1} → ℓ{i+1}',color='#fff0de');arrow(ax,(x+.9,1.2),(x+.9,2.1));arrow(ax,(x+.9,2.8),(x+.9,3.6));arrow(ax,(1.5 if i==0 else x-1.1,2.45),(x,2.45));ax.text(x+1.08,1.6,'U, b',ha='left',fontsize=10);ax.text(x+1.08,3.13,'v, c',ha='left',fontsize=10);ax.text((1.5+x)/2 if i==0 else x-.55,2.75,'W',ha='center',fontsize=10)
    ax.text(5.5,4.65,'时间展开：节点不同，参数 U、W、b、v、c 是同一组',ha='center',weight='bold');ax.text(5.5,.05,'列向量：x_t 维度 D，h_t 维度 H；批次代码 X 为 [N,T,D]，状态为 [N,T,H]',ha='center',fontsize=10)
    save(fig,'01_unrolled.png')
    valid=[(i,t) for i in range(2) for t in range(h['lengths'][i])];labels=[f'{"AB"[i]}{t+1}' for i,t in valid]
    fig,axs=plt.subplots(1,2,figsize=(11,4.5));fig.subplots_adjust(bottom=.24,wspace=.28)
    for key,name,color in [('a','a 前激活','#2166ac'),('z','z logit','#c76b25')]:axs[0].plot(labels,[np.array(r[key])[i,t].item() for i,t in valid],'-o',label=name,color=color)
    axs[0].plot(labels,[r['h'][i][t+1][0] for i,t in valid],'-s',label='h 隐状态',color='#168575');axs[0].axhline(0,color='#ccc',lw=1);axs[0].set(title='同一手算例的逐时间前向',ylabel='数值')
    axs[1].plot(labels,[r['prob'][i][t] for i,t in valid],'-o',color='#2166ac',label='预测概率 p');axs[1].scatter(labels,[h['y'][i][t] for i,t in valid],marker='x',s=65,color='#c76b25',label='目标 y');axs[1].set(title='概率与标签',ylim=(-.1,1.1));handles,labs=[],[]
    for ax in axs:
        hh,ll=ax.get_legend_handles_labels();handles+=hh;labs+=ll;ax.grid(alpha=.15)
    legend(fig,handles,labs,3);save(fig,'02_hand_forward.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5));fig.subplots_adjust(bottom=.24,wspace=.28);xx=np.arange(5)
    for off,key,lab,col in [(-.22,'direct','当前输出路径','#2166ac'),(0,'future','未来时间路径','#c76b25'),(.22,'dh','两条路径相加','#168575')]:axs[0].bar(xx+off,[r[key][i][t][0] for i,t in valid],.22,label=lab,color=col)
    axs[0].set(xticks=xx,xticklabels=labels,title='对 h 的总梯度 = 直接 + 未来');axs[0].axhline(0,color='gray',lw=.6)
    axs[1].plot(labels,[1-r['h'][i][t+1][0]**2 for i,t in valid],'-o',color='#168575',label='tanh 局部导数');axs[1].plot(labels,[r['delta'][i][t][0] for i,t in valid],'-s',color='#7b3294',label='δ = 对 a 的梯度');axs[1].set(title='再乘局部导数得到 δ');axs[1].axhline(0,color='gray',lw=.6)
    handles,labs=[],[]
    for ax in axs:hh,ll=ax.get_legend_handles_labels();handles+=hh;labs+=ll
    legend(fig,handles,labs,3);save(fig,'03_local_backward.png')
    vals=np.array([[np.array(r['contributions'][k])[i,t].item() for k in ['U','W','b','v','c']] for i,t in valid])
    fig,ax=plt.subplots(figsize=(10,5));fig.subplots_adjust(bottom=.12,right=.87);im=ax.imshow(np.vstack([vals,vals.sum(0)]),cmap='RdBu_r',vmin=-.22,vmax=.22,aspect='auto');ax.set(xticks=range(5),xticklabels=['U','W','b','v','c'],yticks=range(6),yticklabels=labels+['总和'],title='共享参数梯度：每个使用位置作贡献，再沿样本和时间求和')
    for i in range(6):
        for j in range(5):ax.text(j,i,f'{np.vstack([vals,vals.sum(0)])[i,j]:+.6f}',ha='center',va='center',color='white' if abs(np.vstack([vals,vals.sum(0)])[i,j])>.15 else '#17212a',fontsize=10)
    fig.colorbar(im,ax=ax,fraction=.04,pad=.03,label='梯度贡献，已含 1/5');save(fig,'04_shared_gradients.png')
    fig,axs=plt.subplots(1,2,figsize=(10,4.5));fig.subplots_adjust(bottom=.24,wspace=.28)
    names=['U','W','b','v','c'];old=[np.array(h['initial_parameters'][k]).item() for k in names];new=[np.array(h['updated_parameters'][k]).item() for k in names];x0=np.arange(5)
    axs[0].bar(x0-.16,old,.32,label='旧参数',color='#2166ac');axs[0].bar(x0+.16,new,.32,label='同步 SGD 后',color='#168575');axs[0].set(xticks=x0,xticklabels=names,title='θ新 = θ旧 − 0.2 × 总梯度')
    axs[1].plot(labels,[r['prob'][i][t] for i,t in valid],'-o',label='旧参数前向',color='#2166ac');axs[1].plot(labels,[h['next_forward']['prob'][i][t] for i,t in valid],'-s',label='新参数下一次前向',color='#168575');axs[1].set(title=f'损失 {r["loss"]:.6f} → {h["next_forward"]["loss"]:.6f}',ylabel='预测概率',ylim=(.3,.65))
    handles,labs=[],[]
    for ax in axs:hh,ll=ax.get_legend_handles_labels();handles+=hh;labs+=ll
    legend(fig,handles,labs,2);save(fig,'05_update.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4.6));fig.subplots_adjust(bottom=.25,wspace=.3);steps=np.arange(1,31)
    for q,col in [(.5,'#2166ac'),(.9,'#168575'),(1.1,'#c76b25'),(1.5,'#9a2876')]:axs[0].semilogy(steps,q**steps,label=f'常数 Jacobian {q}',color=col)
    axs[0].set(title='标量机制示意：不是实验测量',xlabel='连乘次数',ylabel='绝对敏感度')
    a=np.linspace(-4,4,301);axs[1].plot(a,1-np.tanh(a)**2,color='#168575',label='1 − tanh²(a)');axs[1].plot(a,1.5*(1-np.tanh(a)**2),color='#9a2876',label='W=1.5 时的 |J|');axs[1].axhline(1,color='gray',lw=1,ls=':');axs[1].set(title='W 大于 1 仍可因饱和而缩小梯度',xlabel='a',ylabel='局部导数或 Jacobian')
    handles,labs=[],[]
    for ax in axs:hh,ll=ax.get_legend_handles_labels();handles+=hh;labs+=ll;ax.grid(alpha=.2)
    legend(fig,handles,labs,3);save(fig,'06_jacobian.png')
    fig,ax=plt.subplots(figsize=(11,4));ax.axis('off');ax.set(xlim=(0,11),ylim=(0,4))
    for t in range(9):
        x=.3+t*1.15;box(ax,(x,2.1),str(t+1),.65,.6,color='#e5f2ef' if t==8 else '#e9eff6')
        if t:arrow(ax,(x-.5,2.4),(x,2.4))
    for t in [4,8]:
        x=.12+t*1.15;ax.plot([x,x],[1.6,3.1],'--',color='#bc4328',lw=2);ax.text(x,3.35,'detach',ha='center',color='#bc4328')
    ax.text(5.4,.9,'前向：数值状态仍可跨边界传递。反向：最后时刻的损失只能走到第 9 步。',ha='center');ax.text(5.4,.4,'这套分块边界为 1–4、5–8、9；K=4 不保证每个终点都有 4 步反传。',ha='center');ax.text(5.4,3.85,'截断位置与损失位置必须一起标注',ha='center',weight='bold');save(fig,'07_detach.png')
    fig,axs=plt.subplots(2,1,figsize=(11,5));fig.subplots_adjust(hspace=.55,right=.88,bottom=.14)
    for i,ax in enumerate(axs):
        idx=int(np.flatnonzero(d['d8_train_y']==i)[0]);seq=d['d8_train_x'][idx];im=ax.imshow(seq.T,aspect='auto',cmap='RdBu_r',vmin=-1,vmax=1);ax.set(yticks=[0,1,2],yticklabels=['值 / 干扰','写标志','读标志'],xticks=range(9),xticklabels=range(1,10),title=f'独立样本 {idx}：目标 y={int(d["d8_train_y"][idx])}，第 1 步给出符号，第 9 步回答')
    fig.colorbar(im,ax=axs,fraction=.025,pad=.03,label='输入值');axs[-1].set_xlabel('时间步（延迟 D=8，序列长度 T=9）');save(fig,'08_data.png')
    fig,axs=plt.subplots(1,3,figsize=(12,4.6));fig.subplots_adjust(bottom=.25,wspace=.28);styles={7:'-',19:'--',41:':'}
    for ax,delay in zip(axs,[2,8,24]):
        for run in runs:
            if run['delay']!=delay:continue
            ax.semilogy(range(1,len(run['history'])+1),[v['loss_before_update'] for v in run['history']],color=C[run['condition']],ls=styles[run['seed']],lw=1.2,alpha=.85)
        ax.set(title=f'延迟 D={delay}',xlabel='SGD 更新前的步数',ylabel='训练 BCE');ax.grid(alpha=.18)
    handles=[plt.Line2D([0],[0],color=C[k],label=L[k]) for k in C]+[plt.Line2D([0],[0],color='#444',ls=styles[s],label=f'seed {s}') for s in styles]
    legend(fig,handles,[x.get_label() for x in handles],3);save(fig,'09_training.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4.8));fig.subplots_adjust(bottom=.26,wspace=.3)
    for ax,metric in zip(axs,['accuracy','bce']):
        for k,cond in enumerate(C):
            for i,delay in enumerate([2,8,24]):
                vv=[v['final']['test'][metric] for v in runs if v['delay']==delay and v['condition']==cond];xx=i+(k-1)*.22
                ax.scatter(np.array([-.035,0,.035])+xx,vv,color=C[cond],s=30,zorder=3)
                ax.errorbar(xx,np.mean(vv),yerr=np.std(vv,ddof=1) if metric=='accuracy' else None,color=C[cond],fmt='_',ms=13,capsize=4,lw=1.5)
        ax.set(xticks=[0,1,2],xticklabels=['2','8','24'],xlabel='延迟 D',title='封存测试集准确率' if metric=='accuracy' else '封存测试集 BCE');ax.grid(alpha=.15)
        if metric=='accuracy':ax.set_ylim(.35,1.07);ax.axhline(.5,color='gray',ls=':',lw=1)
        else:ax.set_yscale('log')
    handles=[plt.Line2D([0],[0],color=C[k],marker='o',label=L[k]) for k in C];legend(fig,handles,[x.get_label() for x in handles]);fig.text(.5,.12,'圆点为各 seed，横线为均值；准确率误差棒为样本 SD（非置信区间），BCE 不画跨零 SD。',ha='center',fontsize=10);save(fig,'10_test_results.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4.8));fig.subplots_adjust(bottom=.27,wspace=.28)
    for cond in C:
        run=next(v for v in runs if v['delay']==24 and v['seed']==7 and v['condition']==cond)
        axs[0].semilogy(range(1,321),[v['grad_norm_before'] for v in run['history']],color=C[cond],label=L[cond])
    axs[0].axhline(.25,color='black',ls=':',lw=1);axs[0].set(title='D=24、seed 7 的裁剪前范数',xlabel='更新步',ylabel='全参数梯度 L2 范数')
    for k,cond in enumerate(C):
        vals=[next(v for v in runs if v['delay']==24 and v['seed']==s and v['condition']==cond)['clip_fraction'] for s in [7,19,41]]
        axs[1].bar(np.arange(3)+(k-1)*.23,vals,.23,color=C[cond],label=L[cond])
    axs[1].set(xticks=range(3),xticklabels=['7','19','41'],xlabel='seed',ylabel='实际触发比例',title='D=24 的裁剪记录（未裁剪组记 0）');legend(fig,*axs[0].get_legend_handles_labels());save(fig,'11_clipping.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4.7));fig.subplots_adjust(bottom=.30,wspace=.3)
    for cond in C:
        run=next(v for v in runs if v['delay']==24 and v['seed']==7 and v['condition']==cond)
        for ax,key in zip(axs,['probe_final_full','probe_final_training_graph']):
            vals=np.array(run[key]);ax.semilogy(range(1,26),np.maximum(vals,1e-18),'-o',markersize=3,color=C[cond],label=L[cond]);ax.set(xlabel='输入时间步',ylabel='最终 logit 的输入梯度 L2 范数');ax.grid(alpha=.15)
    axs[0].set_title('最终参数：完整计算图敏感度');axs[1].set_title('相同参数：实际训练图敏感度');fig.text(.5,.14,'右图显示在 1e-18 的截断点为精确 0；显示下限不代表非零梯度。',ha='center',fontsize=10)
    legend(fig,*axs[0].get_legend_handles_labels());save(fig,'12_input_gradients.png')
    cf=json.loads((ROOT/'outputs/counterfactual.json').read_text())['runs'];fig,ax=plt.subplots(figsize=(11,4.5));fig.subplots_adjust(bottom=.29)
    for k,cond in enumerate(C):
        group=[v for v in runs if v['delay']==24 and v['condition']==cond]
        for off,key,marker in [(-.05,'original','o'),(.05,'flipped_signal_and_label','x')]:
            vals=[v['final']['test']['accuracy'] if key=='original' else next(a for a in cf if a['id']==v['id'])[key]['accuracy'] for v in group]
            ax.scatter(np.arange(3)+4*k+off,vals,color=C[cond],marker=marker,s=55)
    ax.set(xticks=[1,5,9],xticklabels=list(L.values()),ylim=(.35,1.05),ylabel='准确率',title='D=24 事后反事实诊断：相同干扰，翻转首位信号与标签');ax.axhline(.5,color='gray',ls=':');ax.grid(axis='y',alpha=.15)
    hh=[plt.Line2D([0],[0],color='#333',marker=m,linestyle='none') for m in ['o','x']];legend(fig,hh,['原测试集','信号和目标一起翻转'],2);fig.text(.5,.12,'每组三个横向位置依次为 seeds 7、19、41；此诊断未用于选择参数或模型。',ha='center',fontsize=10);save(fig,'13_counterfactual.png')
    print('13 figures generated')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);a=p.parse_args();main(a.out)
