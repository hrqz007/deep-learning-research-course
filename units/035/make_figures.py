"""Original figures computed from experiment outputs; never draw invented curves."""
from pathlib import Path
import argparse,csv,gzip,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
# Register the bundled system CJK collection explicitly; fontconfig may be uncached.
CJK=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if CJK.exists():
    font_manager.fontManager.addfont(str(CJK))
    CJK_NAME=font_manager.FontProperties(fname=str(CJK)).get_name()
else:
    CJK_NAME='sans-serif'
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
BASE=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':[CJK_NAME,'DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':180})
COL=['#536ba8','#db8a2b','#009e83','#c75067'];SCALES=[.5,1.,math.sqrt(2),2.]
def readcsv(path):
    with path.open() as f:return list(csv.DictReader(f))
def main(source=None,destination=None):
    source=Path(source) if source else BASE/'outputs';dest=Path(destination) if destination else BASE/'figures'
    result=json.loads((source/'results.json').read_text());stats=readcsv(source/'layer_statistics.csv');curves=readcsv(source/'training_curves.csv');width=readcsv(source/'width_study.csv');dis=json.loads(gzip.decompress((source/'selected_distributions.json.gz').read_bytes()).decode('utf-8'))
    if len(stats)!=576 or len(result['cases'])!=48:raise ValueError('expected full default protocol before plotting')
    # Validate every input used by any figure before opening an output file.
    expected={(a,s,k,l) for a in ['linear','tanh','relu','sigmoid'] for s in SCALES for k in (35,36,37) for l in range(1,13)}
    observed=set()
    required={'h_second','h_variance','mean_unit_data_variance','variance_unit_data_means','probe_grad_second'}
    for row in stats:
        if not required.issubset(row):raise ValueError('missing figure statistic')
        key=(row['activation'],float(row['alpha']),int(row['seed']),int(row['layer']))
        if key in observed:raise ValueError('duplicate layer row')
        observed.add(key)
        for k,v in row.items():
            if k!='activation' and not math.isfinite(float(v)):raise ValueError('nonfinite figure input')
    if observed!=expected:raise ValueError('incomplete layer grid')
    for row in curves:
        for k in ('alpha','seed','step','loss'):
            if not math.isfinite(float(row[k])):raise ValueError('nonfinite training curve')
        if float(row['loss'])<=0:raise ValueError('log plot requires positive recorded loss')
    for row in width:
        for v in row.values():
            if not math.isfinite(float(v)):raise ValueError('nonfinite width input')
    if len(width)!=48:raise ValueError('expected 48 width observations')
    for a in ('linear','tanh','relu','sigmoid'):
        for scale in (1.,math.sqrt(2),2.):
            group=dis[f'{a}_{scale:.6f}']
            for prefix in ('z','h','task_grad','probe_grad'):
                for layer in (1,6,12):
                    values=np.asarray(group[f'{prefix}_{layer}'],dtype=float)
                    if values.shape!=(8192,) or not np.isfinite(values).all():raise ValueError('invalid distribution')
    def finite_tree(value):
        if isinstance(value,dict):
            for v in value.values():finite_tree(v)
        elif isinstance(value,list):
            for v in value:finite_tree(v)
        elif isinstance(value,(int,float)) and not math.isfinite(value):raise ValueError('nonfinite result')
    finite_tree(result)
    casekeys={(c['activation'],c['alpha'],c['seed']) for c in result['cases']}
    if casekeys!={(a,s,k) for a in ['linear','tanh','relu','sigmoid'] for s in SCALES for k in (35,36,37)}:raise ValueError('incomplete case grid')
    protocol=result['protocol']
    for key,value in {'n':128,'input_features':16,'depth':12,'width':64,'steps':100,'lr':.03}.items():
        if protocol.get(key)!=value:raise ValueError('figure protocol differs from default')
    if len({(int(r['width']),int(r['seed'])) for r in width})!=48 or {(int(r['width']),int(r['seed'])) for r in width}!={(w,k) for w in (8,32,128) for k in range(40,56)}:raise ValueError('incomplete width grid')
    if any(float(r['last_second_over_input'])<=0 for r in width):raise ValueError('log width plot requires positive moments')
    curvegroups={key:[] for key in casekeys}
    for row in curves:
        key=(row['activation'],float(row['alpha']),int(row['seed']))
        if key not in curvegroups:raise ValueError('unexpected curve configuration')
        step=int(row['step'])
        if str(step)!=row['step'] or step<0 or step>100:raise ValueError('invalid recorded step')
        curvegroups[key].append((step,float(row['loss'])))
    for case in result['cases']:
        key=(case['activation'],case['alpha'],case['seed']);points=curvegroups[key]
        if not points or [p[0] for p in points]!=list(range(len(points))):raise ValueError('curve must start at 0 and remain contiguous')
        if case['status'] not in ('completed','stopped_loss_limit','stopped_nonfinite_update'):raise ValueError('invalid stop status')
        if case['status']=='completed' and len(points)!=101:raise ValueError('completed curve must include step100')
        if not isinstance(case['updates_applied'],int) or not points[-1][0]<=case['updates_applied']<=100:raise ValueError('invalid update count')
        if points[0][1]!=case['initial_loss'] or points[-1][1]!=case['last_recorded_loss']:raise ValueError('curve summary mismatch')
    # Pre-access nested mechanisms and hand fields so missing-schema errors are pre-write too.
    for key in ('loss','new_loss'):
        float(result['hand'][key]['float'])
    if len(result['hand']['gradient'])!=13:raise ValueError('expected 13 hand gradients')
    for value in result['hand']['gradient']:float(value['float'])
    mm=result['mechanisms']['relu_moments']
    for obj,key in [('z','second'),('relu','second'),('relu','variance'),('relu','mean')]:float(mm[obj][key])
    for key in ('diagonal_only','actual_variance'):float(result['mechanisms']['correlated_input'][key])
    dest.mkdir(parents=True,exist_ok=True)
    def save(fig,name):fig.savefig(dest/name,bbox_inches='tight',facecolor='white');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.1));ax.set(xlim=(-.1,10.4),ylim=(0,3));ax.axis('off')
    boxes=[(.1,'输入 X\n2 × 1'),(2.05,'仿射 + ReLU\nZ₁,H₁: 2 × 2'),(4.3,'仿射 + ReLU\nZ₂,H₂: 2 × 2'),(6.6,'线性输出\nŷ: 2 × 1'),(8.5,'half-MSE\nL: 标量')]
    for j,(x,label) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x,1.1),1.6,1,boxstyle='round,pad=.1',facecolor=['#dbe8f3','#d4eee7','#d4eee7','#fff0d7','#f3dbe1'][j],edgecolor='#64748b'));ax.text(x+.8,1.6,label,ha='center',va='center')
        if j:ax.annotate('',(x-.1,1.8),(boxes[j-1][0]+1.7,1.8),arrowprops=dict(arrowstyle='->',color='#276b87',lw=2))
    ax.annotate('梯度：每条局部路径相乘，汇合处相加',xy=(.9,.5),xytext=(6.3,.5),ha='center',arrowprops=dict(arrowstyle='->',color=COL[3],lw=2),color=COL[3]);ax.set_title('同一批样本的一次前向与反向',loc='left',weight='bold');save(fig,'01_computation_path.png')
    r=result['hand'];fig,axs=plt.subplots(1,3,figsize=(10,3))
    axs[0].bar(['更新前','更新后'],[r['loss']['float'],r['new_loss']['float']],color=[COL[0],COL[2]]);axs[0].set_ylabel('half-MSE');axs[0].set_title('η = 0.1 的实际重算')
    g=[v['float'] for v in r['gradient']];axs[1].bar(range(13),g,color=[COL[0] if v>=0 else COL[3] for v in g]);axs[1].axhline(0,color='gray',lw=.7);axs[1].set_xticks(range(0,13,2));axs[1].set(xlabel='参数展平索引 0–12',ylabel='∂L/∂θ',title='13 个参数均有明确梯度')
    gate=np.array([[1,0,1,1],[0,1,0,1]]);axs[2].imshow(gate,cmap='YlGn',vmin=0,vmax=1);axs[2].set(xticks=range(4),xticklabels=['Z₁a','Z₁b','Z₂a','Z₂b'],yticks=[0,1],yticklabels=['x=1','x=-1'],title='手算点的 ReLU 门')
    for (i,j),v in np.ndenumerate(gate):axs[2].text(j,i,str(v),ha='center',va='center',color='white' if v else 'black')
    fig.tight_layout();save(fig,'02_hand_step.png')
    z=np.linspace(-5,5,1001);sig=1/(1+np.exp(-z));fig,axs=plt.subplots(1,2,figsize=(10,3.5))
    for label,h,d,c in [('linear',z,np.ones_like(z),COL[0]),('tanh',np.tanh(z),1-np.tanh(z)**2,COL[1]),('ReLU',np.maximum(z,0),(z>0).astype(float),COL[2]),('sigmoid',sig,sig*(1-sig),COL[3])]:
        axs[0].plot(z,h,label=label,color=c);axs[1].plot(z,d,label=label,color=c)
    for ax in axs:ax.axvline(0,color='gray',lw=.5);ax.set_xlabel('激活前 z');ax.legend(fontsize=8)
    axs[0].set(ylabel='φ(z)',ylim=(-1.3,2.3),title='函数值（线性/ReLU 大值在图外）');axs[1].set(ylabel='φ′(z)',ylim=(-.03,1.1),title='ReLU 的 z=0 导数是实现约定');fig.tight_layout();save(fig,'03_activation_derivatives.png')
    rng=np.random.default_rng(351);a=rng.standard_normal(100000);z=np.r_[a,-a];h=np.maximum(z,0);m=result['mechanisms']['relu_moments'];fig,axs=plt.subplots(1,2,figsize=(10,3.5))
    axs[0].hist(z,bins=80,density=True,alpha=.65,color=COL[0],label='对称 Z');axs[0].hist(h,bins=80,density=True,alpha=.55,color=COL[2],label='ReLU(Z)');axs[0].set(ylim=(0,1.3),xlabel='值',ylabel='密度',title='0 处约一半质量（柱高超出上界）');axs[0].legend()
    vals=[m['z']['second'],m['relu']['second'],m['relu']['variance'],m['relu']['mean']**2];axs[1].bar(['E[Z²]','E[H²]','Var(H)','E[H]²'],vals,color=[COL[0],COL[2],COL[1],COL[3]]);axs[1].set(title='二阶矩 = 中心方差 + 均值平方',ylabel='样本统计量');fig.tight_layout();save(fig,'04_relu_moments.png')
    def group(act,alpha,key):
        return np.array([[float(r[key]) for r in stats if r['activation']==act and float(r['alpha'])==alpha and int(r['seed'])==seed] for seed in (35,36,37)])
    for key,name,title in [('h_second','06_forward_depth.png','激活二阶矩 Ê[H²]'),('probe_grad_second','07_backward_depth.png','探针梯度二阶矩 / 第12层值')]:
        fig,axs=plt.subplots(2,2,figsize=(10,6.2))
        for ax,act in zip(axs.flat,['linear','tanh','relu','sigmoid']):
            for alpha,c in zip(SCALES,COL):
                arr=group(act,alpha,key)
                if key=='probe_grad_second':arr=arr/arr[:,-1:]
                med=np.median(arr,0);ax.plot(range(1,13),med,color=c,label=f'α={alpha:.3g}');ax.fill_between(range(1,13),arr.min(0),arr.max(0),color=c,alpha=.13)
                if key=='h_second' and act in ('linear','relu'):
                    pred=(alpha**2/(2 if act=='relu' else 1))**np.arange(1,13);ax.plot(range(1,13),pred,color=c,ls=':',lw=.9)
            ax.set(yscale='log',xlabel='层号 l',ylabel=title,title=act);ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.15)
        fig.tight_layout();save(fig,name)
    fig,axs=plt.subplots(2,2,figsize=(10,6.2))
    for row,(act,alpha) in enumerate([('relu',math.sqrt(2)),('tanh',2.)]):
        arr=dis[f'{act}_{alpha:.6f}']
        for layer,c in zip((1,6,12),COL):
            h=np.array(arr[f'h_{layer}']);g=np.array(arr[f'task_grad_{layer}']);nz=g[g!=0]
            axs[row,0].hist(h,bins=60,density=True,histtype='step',color=c,label=f'层{layer}')
            axs[row,1].hist(np.log10(np.abs(nz)),bins=50,density=True,histtype='step',color=c,label=f'层{layer}; 零={np.mean(g==0):.0%}')
        axs[row,0].set(title=f'{act}, α={alpha:.3g}, seed=35',xlabel='激活 H',ylabel='密度')
        axs[row,1].set(title='任务梯度：非零部分的分布',xlabel='log₁₀ |∂L/∂Z|',ylabel='非零条件密度')
        for ax in axs[row]:ax.legend(fontsize=8)
    fig.tight_layout();save(fig,'08_distributions.png')
    fig,axs=plt.subplots(2,2,figsize=(10,6.4))
    for ax,act in zip(axs.flat,['linear','tanh','relu','sigmoid']):
        for alpha,c in zip(SCALES,COL):
            for seed in (35,36,37):
                rows=[r for r in curves if r['activation']==act and float(r['alpha'])==alpha and int(r['seed'])==seed]
                xx=[int(r['step']) for r in rows];yy=[float(r['loss']) for r in rows]
                ax.plot(xx,yy,color=c,alpha=.75 if seed==35 else .3,label=f'α={alpha:.3g}' if seed==35 else None)
                case=next(v for v in result['cases'] if v['activation']==act and v['alpha']==alpha and v['seed']==seed)
                if case['status']!='completed':ax.scatter(xx[-1],yy[-1],marker='x',color=c,s=45)
        ax.set(yscale='log',xlabel='已完成更新次数',ylabel='训练 half-MSE',title=act,xlim=(-2,102));ax.grid(alpha=.15);ax.legend(ncol=2,fontsize=7)
    fig.tight_layout();save(fig,'09_training_curves.png')
    fig,axs=plt.subplots(1,2,figsize=(10,3.5));values=[]
    for k,w in enumerate((8,32,128)):
        vals=np.array([float(r['last_second_over_input']) for r in width if int(r['width'])==w]);values.append(vals);axs[0].scatter(np.full(16,k)+np.linspace(-.13,.13,16),vals,s=20,color=COL[k])
    axs[0].boxplot(values,positions=range(3),showfliers=False);axs[0].set(xticks=range(3),xticklabels=['8','32','128'],xlabel='宽度',ylabel='第12层二阶矩 / 输入二阶矩',yscale='log',title='16 个初始化种子，全部展示');axs[0].axhline(1,ls=':',color='gray')
    arr=group('relu',math.sqrt(2),'h_variance')[0];within=group('relu',math.sqrt(2),'mean_unit_data_variance')[0];between=group('relu',math.sqrt(2),'variance_unit_data_means')[0]
    axs[1].stackplot(range(1,13),within,between,colors=[COL[0],COL[1]],labels=['各单元数据方差的平均','单元数据均值的方差']);axs[1].plot(range(1,13),arr,color='black',lw=1,label='池化中心方差');axs[1].set(xlabel='层号',ylabel='中心方差',title='同一网络中，统计轴也改变问题');axs[1].legend(fontsize=8);fig.tight_layout();save(fig,'10_width_and_axes.png')
    fig,axs=plt.subplots(1,3,figsize=(10,3.2));v=result['mechanisms'];axs[0].bar(['忽略协方差','完整计算'],[v['correlated_input']['diagonal_only'],v['correlated_input']['actual_variance']],color=[COL[1],COL[0]]);axs[0].set(ylabel='固定 W 的输出方差',title='X₁ = X₂ 的反例')
    axs[1].bar(['假定独立','实际联合'],[.25,.5],color=[COL[1],COL[3]]);axs[1].set(ylabel='E[G²T²]',title='门 G 与上游量 T 相同')
    xx=np.arange(2);axs[2].bar(xx-.15,[1,1],width=.3,label='I₂',color=COL[2]);axs[2].bar(xx+.15,[math.sqrt(2),0],width=.3,label='diag(√2,0)',color=COL[0]);axs[2].set(xticks=xx,xticklabels=['σ₁','σ₂'],ylabel='奇异值',title='平均平方相同，方向不同');axs[2].legend(fontsize=8);fig.tight_layout();save(fig,'11_assumption_failures.png')
    fig,ax=plt.subplots(figsize=(9,3.1));ax.axis('off')
    ax.text(.02,.84,'W 的布局：[输出单元数, 输入特征数]',fontsize=15,weight='bold');ax.text(.03,.48,'W: [64, 16]\nfan_in = 16\nfan_out = 64',fontsize=12,va='center',bbox=dict(boxstyle='round',facecolor='#dbe8f3',edgecolor='none'))
    ax.text(.34,.54,'向前：每个输出累加 16 项\nReLU 二阶矩系数 = 16s² / 2',fontsize=12,color=COL[0]);ax.text(.34,.20,'向后：每个输入接收 64 路\n近似梯度系数 = 64s² / 2',fontsize=12,color=COL[3]);save(fig,'05_fan_layout.png')
    print(json.dumps({'figures':len(list(dest.glob('*.png'))),'source_cases':48}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--destination',type=Path);a=p.parse_args();main(a.source,a.destination)
