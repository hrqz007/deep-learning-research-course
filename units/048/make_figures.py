"""Original diagrams and measured plots. Legends always outside the data axes."""
from pathlib import Path
import argparse,json,tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager
try:font_manager.findfont('Noto Sans CJK SC',fallback_to_default=False)
except ValueError:
    from fontTools.ttLib import TTCollection
    p=Path(tempfile.gettempdir())/'048-font.ttf'
    if not p.exists():
        c=TTCollection('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');next(f for f in c.fonts if f['name'].getDebugName(1)=='Noto Sans CJK SC').save(p)
    font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':['Noto Sans CJK SC','DejaVu Sans'],'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.unicode_minus':False})
HERE=Path(__file__).resolve().parent
NAMES={'plain':'原始 CNN','brightness':'亮度增强','random_stamp':'角标随机化','mask_stamp':'已知位置遮蔽'}
COLORS=['#2464a1','#d07827','#25836a','#9a549d']
def save(fig,out,name):fig.savefig(out/name,bbox_inches='tight',facecolor='white');plt.close(fig)
def box(ax,x,y,w,h,s):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02',facecolor='#eaf1f7',edgecolor='#527c9e'));ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=11)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#64748b'))
def main(results=HERE/'outputs/results.json',output=HERE/'figures'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);r=json.loads(Path(results).read_text());runs=r['runs'];final=r['finalists'];methods=r['protocol']['methods']
    fig,ax=plt.subplots(figsize=(11.4,3.5));ax.axis('off');ax.set(xlim=(0,12),ylim=(0,4))
    for x,s in [(0.1,'写任务与数据许可\n固定预算和选择规则'),(3.15,'训练 24 个候选\n只看两个开发集'),(6.2,'保存选择与权重哈希\n方案冻结'),(9.25,'打开两个封存测试\n报告分组和失败')]:box(ax,x,1.2,2.5,1.2,s)
    for x in (2.65,5.7,8.75):arrow(ax,(x,1.8),(x+.45,1.8))
    ax.text(6,.35,'冻结后产生的新想法只能进入下一项目；本轮测试不能再次担任开发集',ha='center');save(fig,out,'01_evidence_flow.png')
    fig,axs=plt.subplots(2,6,figsize=(11,4))
    x=np.load(HERE/'data/train_x.npy');meta=json.loads((HERE/'data/train_meta.json').read_text())
    for i,ax in enumerate(axs.flat):
        ax.imshow(x[i,0],cmap='gray',vmin=0,vmax=1);m=meta[i];ax.set_title(f"{i}: {'环' if m['label'] else '盘'} / 角标{m['stamp']}",fontsize=9);ax.axis('off')
    fig.suptitle('原创 20×20 图像：中心形状决定标签，左上角标只相关');fig.tight_layout();save(fig,out,'02_data_samples.png')
    fig,ax=plt.subplots(figsize=(10,3.8));manifest=json.loads((HERE/'data/manifest.json').read_text());splits=list(manifest['splits']);ax.bar(range(5),[manifest['splits'][s]['stamp_match_actual'] for s in splits],color=[COLORS[0]]*3+[COLORS[1]]*2);ax.set(xticks=range(5),xticklabels=['训练','开发同域','开发偏移','封存同域','封存反相关'],ylim=(0,1.12),ylabel='角标与标签相同的比例',title='开发偏移域不能替代最终反相关测试域')
    for i,s in enumerate(splits):ax.text(i,manifest['splits'][s]['stamp_match_actual']+.025,f"n={manifest['splits'][s]['n']}",ha='center',fontsize=10)
    fig.tight_layout();save(fig,out,'03_split_domains.png')
    fig,ax=plt.subplots(figsize=(11,3.8));ax.axis('off');ax.set(xlim=(0,12),ylim=(0,4))
    for x,w,s in [(.1,2,'A: [0,1], y=0\nB: [1,2], y=1'),(3,2.1,'共享 1×1 卷积\nh = ReLU(wx+b)'),(6.1,1.7,'空间平均\nm = (h₁+h₂)/2'),(9,2.6,'logit = am+c\np = sigmoid(logit)')]:box(ax,x,1.3,w,1.2,s)
    for a,b in [((2.15,1.9),(2.95,1.9)),((5.15,1.9),(6.05,1.9)),((7.85,1.9),(8.95,1.9))]:arrow(ax,a,b)
    ax.text(6,.45,'四个参数被两个样本和两个空间位置共享；每条反向路径都必须相加',ha='center');save(fig,out,'04_hand_graph.png')
    h=r['hand']['rounds'];fig,ax=plt.subplots(figsize=(8.8,3.8));v=np.array([q['gradient'] for q in h[0]['rows']]+[h[0]['gradient']]);mx=np.abs(v).max();im=ax.imshow(v,cmap='RdBu_r',vmin=-mx,vmax=mx,aspect='auto')
    for (i,j),z in np.ndenumerate(v):ax.text(j,i,f'{z:+.6f}',ha='center',va='center',color='white' if abs(z)>.65*mx else '#152638',fontsize=12)
    ax.set(xticks=range(4),xticklabels=['w','b','a','c'],yticks=range(3),yticklabels=['样本 A 已含1/2','样本 B 已含1/2','相加得批梯度'],title='梯度账本可见样本贡献方向相反');fig.colorbar(im,ax=ax,label='偏导');fig.tight_layout();save(fig,out,'05_hand_gradients.png')
    fig,axs=plt.subplots(1,2,figsize=(10.5,3.5));axs[0].plot(range(3),[q['loss'] for q in h],'o-',color=COLORS[0]);axs[0].set(xlabel='同步更新次数',ylabel='平均二元交叉熵',xticks=range(3));handles=[]
    for i,label in enumerate(['A 标签0','B 标签1']):handles+=axs[1].plot(range(3),[q['rows'][i]['p'] for q in h],'o-',label=label)
    axs[1].set(xlabel='同步更新次数',ylabel='预测为1的概率',xticks=range(3),ylim=(.4,.8));fig.legend(handles,[h.get_label() for h in handles],loc='lower center',ncol=2,bbox_to_anchor=(.5,-.02));fig.tight_layout(rect=(0,.1,1,1));save(fig,out,'06_hand_updates.png')
    fig,axs=plt.subplots(2,2,figsize=(10.6,6.8));handles=[]
    for ax,method,color in zip(axs.flat,methods,COLORS):
        for q in [v for v in runs if v['method']==method]:
            line,=ax.plot([v['epoch'] for v in q['history']],[(v['dev_iid']['loss']+v['dev_shift']['loss'])/2 for v in q['history']],color=color,ls='-' if q['lr']==.03 else '--',alpha=.65,lw=1.5)
        ax.set(title=NAMES[method],xlabel='epoch',ylabel='两开发域平均交叉熵');ax.grid(alpha=.2)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([0],[0],color='#444',ls='-'),Line2D([0],[0],color='#444',ls='--')],['学习率 0.03，三条种子轨迹','学习率 0.1，三条种子轨迹'],loc='lower center',ncol=2,bbox_to_anchor=(.5,-.005));fig.tight_layout(rect=(0,.06,1,1));save(fig,out,'07_all_candidates.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5));handles=[]
    for j,(name,title) in enumerate([('test_iid','封存同域'),('test_shift','封存反相关')]):
        ax=axs[j]
        for i,(method,color) in enumerate(zip(methods,COLORS)):
            vals=[q[name]['accuracy'] for q in final if q['method']==method];ax.bar(i,np.mean(vals),color=color,alpha=.25,width=.62);ax.scatter(np.array([i-.13,i,i+.13]),vals,c=color,s=32,zorder=3)
        ax.set(xticks=range(4),xticklabels=['原始','亮度','随机角标','遮蔽'],ylim=(0,1.07),ylabel='准确率',title=title+'：柱为均值，点为全部种子');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();save(fig,out,'08_test_results.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5));groups=['y0_s0','y0_s1','y1_s0','y1_s1'];handles=[]
    for ax,name,title in zip(axs,['test_iid','test_shift'],['同域分组','反相关分组']):
        for method,color in zip(methods,COLORS):
            vals=[np.mean([q[name]['groups'][g]['error_rate'] for q in final if q['method']==method]) for g in groups];line,=ax.plot(range(4),vals,'o-',color=color,label=NAMES[method]);
            if name=='test_iid':handles.append(line)
        ns=[final[0][name]['groups'][g]['n'] for g in groups];ax.set(xticks=range(4),xticklabels=[g.replace('_',' / ')+f'\nn={n}' for g,n in zip(groups,ns)],ylim=(-.035,1.05),ylabel='三种子平均错误率',title=title)
    fig.legend(handles,[q.get_label() for q in handles],loc='lower center',ncol=4,bbox_to_anchor=(.5,-.02));fig.tight_layout(rect=(0,.11,1,1));save(fig,out,'09_group_errors.png')
    fig,ax=plt.subplots(figsize=(9.4,4));width=.34;xx=np.arange(4)
    for j,(name,label,color) in enumerate([('test_iid','同域',COLORS[0]),('test_shift','反相关',COLORS[1])]):
        vals=[np.mean([q[name]['counterfactual']['flip_rate'] for q in final if q['method']==method]) for method in methods];ax.bar(xx+(j-.5)*width,vals,width,color=color,label=label)
    ax.set(xticks=xx,xticklabels=[NAMES[m] for m in methods],ylabel='只翻转角标后预测改变比例',ylim=(0,1),title='配对干预诊断：中心图像保持原字节');fig.legend(loc='lower center',ncol=2,bbox_to_anchor=(.5,-.02));fig.tight_layout(rect=(0,.12,1,1));save(fig,out,'10_counterfactual.png')
    # Deterministic first six errors of the plain method, lowest preregistered seed.
    chosen=next(q for q in final if q['method']=='plain' and q['seed']==11);bad=chosen['test_shift']['error_indices'][:6];xx=np.load(HERE/'data/test_shift_x.npy');mm=json.loads((HERE/'data/test_shift_meta.json').read_text());fig,axs=plt.subplots(1,6,figsize=(11,2.7))
    for ax,i in zip(axs,bad):
        ax.imshow(xx[i,0],cmap='gray',vmin=0,vmax=1);v=mm[i];ax.set_title(f"#{i} 真{v['label']} 预测{chosen['test_shift']['predictions'][i]}\n角标{v['stamp']} p1={chosen['test_shift']['probabilities'][i][1]:.2f}",fontsize=9);ax.axis('off')
    for ax in axs[len(bad):]:ax.axis('off');ax.text(.5,.5,'无更多错误',ha='center')
    fig.suptitle('失败案例：固定取原始 CNN seed11 反相关测试最前六个错误',fontsize=11);fig.tight_layout();save(fig,out,'11_failure_cases.png')
    return sorted(p.name for p in out.glob('*.png'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=HERE/'outputs/results.json');p.add_argument('--output',type=Path,default=HERE/'figures');a=p.parse_args();print(main(a.results,a.output))
