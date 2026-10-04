"""Original figures. Validate every frozen input before touching the destination."""
from pathlib import Path
import argparse,json,gzip,hashlib,csv,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import experiment as e
ROOT=Path(__file__).resolve().parent
# Filled from the actual six-file default run, never from fabricated curves.
OUTPUT_HASHES = {'final_evaluation.json': 'f08c78e5afa55b00d325d4d4b4735ef5f17fde3b94bfb7c0c494f950055f1a3c', 'hand_chain.json': '37333bdf448070c490093454eb0e3170e9d8c75c0ff24a770d6171f577271cf5', 'results.json': 'ae135c68f6abf5e1b481d1008e2418c4cab4f406be91a5e27ba70947061ed8cf', 'selection_simulation.json': '2243d8bbc8ff90bd268908e6645ca60c656cffe8c7144cb2cb6b2e8fea4d91fa', 'training_curves.csv': '65b474a3a4d6b9f36f248b694acc631e86099562e8ed048e24e48d3080872a0b', 'training_traces.json.gz': 'de2fd923e36ee9bad0bee347feeb4cad63514adfc5db4aae953504f858eed636'}
CJK=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if CJK.exists():font_manager.fontManager.addfont(str(CJK))
FONT=font_manager.FontProperties(fname=str(CJK)).get_name() if CJK.exists() else 'sans-serif'
plt.rcParams.update({'font.family':[FONT,'DejaVu Sans'],'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':185})
COL=['#28618a','#d77930','#12927c','#9a496e','#7766ad','#62717d']

def main(source=None,destination=None):
    e.verify_inputs();src=Path(source) if source else ROOT/'outputs';dest=Path(destination) if destination else ROOT/'figures'
    for name,sha in OUTPUT_HASHES.items():
        if hashlib.sha256((src/name).read_bytes()).hexdigest()!=sha:raise ValueError('changed default figure input: '+name)
    r=json.loads((src/'results.json').read_text());final=json.loads((src/'final_evaluation.json').read_text());hand=json.loads((src/'hand_chain.json').read_text());sim=json.loads((src/'selection_simulation.json').read_text());runs=json.loads(gzip.decompress((src/'training_traces.json.gz').read_bytes()))['runs']
    with (src/'training_curves.csv').open() as f:curves=list(csv.DictReader(f))
    if len(runs)!=51 or len(curves)!=12257:raise ValueError('not the complete default experiment')
    train=e.load_split('train');test=e.load_split('test')
    dest.mkdir(parents=True,exist_ok=True)
    def save(fig,name):fig.savefig(dest/name,bbox_inches='tight',facecolor='white');plt.close(fig)
    def box(ax,x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02',facecolor=color,edgecolor='#718096',lw=.8));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
    fig,ax=plt.subplots(figsize=(9,4.6));ax.set(xlim=(0,10),ylim=(0,5));ax.axis('off')
    box(ax,.3,3.6,2.3,.8,'观察：损失、梯度\n样本、版本、预算','#e1edf6');box(ax,3.4,3.6,2.3,.8,'提出多个解释\n写出可证伪预测','#fff0d9');box(ax,6.6,3.6,2.9,.8,'最小可区分实验\n同一数据、同一种子','#deeee8')
    box(ax,.3,1.6,2.3,1.,'先查数据与计算\n再查优化与函数类','#e1edf6');box(ax,3.4,1.6,2.3,1.,'固定搜索与停止规则\n保留失败及总预算','#fff0d9');box(ax,6.6,1.6,2.9,1.,'冻结选择后才测测试集\n报告未排除的解释','#deeee8')
    for x in [2.7,5.9]:ax.annotate('',xy=(x+.6,4),xytext=(x,4),arrowprops=dict(arrowstyle='->',lw=1.5,color=COL[0]));ax.annotate('',xy=(x+.6,2.1),xytext=(x,2.1),arrowprops=dict(arrowstyle='->',lw=1.5,color=COL[0]))
    ax.text(5,.6,'曲线形状提出问题；受控对照才缩小解释范围。',ha='center',fontsize=14,color=COL[0]);save(fig,'01_diagnosis_evidence.png')
    fig,ax=plt.subplots(figsize=(9,4.1));ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
    for x,text,color in [(.2,'同一输入 x\n-1 或 +1','#e1edf6'),(2.5,'z = wx+b\nh = ReLU(z)\n2个隐藏单元','#deeee8'),(5.0,'预测 = a·h+c\n残差 r = 预测-y','#fff0d9'),(7.6,'单例 ℓ = r²/2\n批平均 L=(ℓ₁+ℓ₂)/2','#f2e1e8')]:box(ax,x,1.7,2.1,1.3,text,color)
    for x in [2.35,4.75,7.25]:ax.annotate('',(x+.15,2.3),(x-.05,2.3),arrowprops=dict(arrowstyle='->',lw=1.8,color=COL[0]))
    ax.text(5,.75,'反向：r/2 → 乘 a → 乘门 → 乘 x；共享参数的两例贡献相加',ha='center',color=COL[3],fontsize=12)
    ax.text(5,3.5,'第1次更新用 θ₀；第2次重新前向并用 θ₁ 求所有局部导数',ha='center',fontsize=12);save(fig,'02_hand_computation.png')
    fig,axs=plt.subplots(1,2,figsize=(9,3.8));loss=[s['loss']['float'] for s in hand['steps']]+[hand['third_loss']['float']]
    axs[0].plot([0,1,2],loss,'o-',color=COL[0]);axs[0].set(xlabel='已完成更新次数',ylabel='两样本 half-MSE',xticks=[0,1,2],title='每次都重新计算的损失')
    for i,l in enumerate(loss):axs[0].annotate(f'{l:.6f}',(i,l),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
    grad=np.array([[v['float'] for v in s['gradient']] for s in hand['steps']]);m=axs[1].imshow(grad,cmap='RdBu_r',vmin=-.25,vmax=.25,aspect='auto');axs[1].set(yticks=[0,1],yticklabels=['第1步','第2步'],xticks=range(7),xticklabels=hand['parameter_order'],title='全部7个梯度随参数变化')
    for (i,j),v in np.ndenumerate(grad):axs[1].text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=8,color='white' if abs(v)>.15 else 'black')
    fig.colorbar(m,ax=axs[1],shrink=.65);fig.tight_layout();save(fig,'03_hand_updates.png')
    fig,axs=plt.subplots(2,3,figsize=(10,6.7));names=[('reference','参照'),('slow','极小步长'),('unstable','过大步长'),('linear','线性函数类'),('shuffled_labels','训练标签置乱'),('strong_penalty','过强惩罚')]
    for ax,(name,title) in zip(axs.flat,names):
        rs=[v for v in runs if v['stage']=='diagnostic' and v['config']['id']==name]
        for j,run in enumerate(rs):
            xx=[v['step'] for v in run['curve']]
            ax.plot(xx,[v['train_data_loss'] for v in run['curve']],color=COL[j],label=f'{run["seed"]} 训练' if name=='reference' else None)
            ax.plot(xx,[v['validation_data_loss'] for v in run['curve']],color=COL[j],ls='--',alpha=.8)
            if run['failure']:ax.plot(xx[-1],run['curve'][-1]['train_data_loss'],'x',color=COL[j],ms=9)
        ax.set(yscale='log',xlabel='已接受的更新数',ylabel='half-MSE',title=title);ax.grid(alpha=.15)
    axs[0,0].legend(fontsize=7);fig.suptitle('实线：训练数据损失；虚线：验证数据损失；×：停止前末状态',fontsize=12);fig.tight_layout();save(fig,'04_diagnostic_curves.png')
    fig,axs=plt.subplots(1,2,figsize=(9,3.8));xx=np.arange(3)
    for label,name,c in [('参照','reference',COL[0]),('输入放大50倍','saturated_input',COL[1])]:
        rs=[v for v in runs if v['stage']=='diagnostic' and v['config']['id']==name]
        axs[0].plot(xx,[v['curve'][0]['saturated_fraction'] for v in rs],'o-',label=label,color=c)
        axs[1].plot(xx,[v['curve'][-1]['validation_data_loss'] for v in rs],'o-',label=label,color=c)
    for ax in axs:ax.set(xticks=xx,xticklabels=['4001','4002','4003'],xlabel='配对初始化种子');ax.legend(fontsize=9)
    axs[0].set(ylabel='初始化 |h|>0.99 的比例',title='确实更饱和');axs[1].set(ylabel='第180步验证 half-MSE',title='但此次验证反而更好');fig.tight_layout();save(fig,'05_saturation_counterexample.png')
    over=next(v for v in runs if v['stage']=='overfit_probe');fig,axs=plt.subplots(1,2,figsize=(9,3.8));xx=[v['step'] for v in over['curve']]
    for key,label,c in [('train_data_loss','16例训练',COL[0]),('validation_data_loss','128例验证',COL[1])]:axs[0].plot(xx,[v[key] for v in over['curve']],label=label,color=c)
    best=r['overfit_probe']['best_validation_step'];axs[0].axvline(best,color=COL[3],ls=':',label=f'验证最小：{best}步');axs[0].set(yscale='log',xlabel='更新次数',ylabel='half-MSE',title='过拟合探针：固定32单元、2500步');axs[0].legend(fontsize=8)
    for run,c in zip([v for v in runs if v['stage']=='tiny_probe'],COL):
        label='完整标签' if run['config']['label_mode']=='intact' else '置乱标签'
        axs[1].plot([v['step'] for v in run['curve']],[v['train_data_loss'] for v in run['curve']],color=c,label=label)
    axs[1].set(yscale='log',xlabel='更新次数',ylabel='8例所用标签的训练 half-MSE',title='小批次探针：保留未完全拟合');axs[1].legend(fontsize=9);fig.tight_layout();save(fig,'06_overfit_and_tiny_probes.png')
    fig,axs=plt.subplots(1,2,figsize=(9,4.));scores=r['search_scores'];grid=np.array([s['mean_validation_half_mse'] for s in scores]).reshape(3,3)
    im=axs[0].imshow(grid,cmap='YlGnBu',aspect='auto');axs[0].set(xticks=range(3),xticklabels=['0','0.001','0.01'],yticks=range(3),yticklabels=['0.01','0.05','0.2'],xlabel='L2系数 λ',ylabel='步长 η',title='3种子均值，越低越好')
    for (i,j),v in np.ndenumerate(grid):axs[0].text(j,i,f'{v:.5f}',ha='center',va='center',fontsize=10,color='white' if v>.066 else 'black')
    fig.colorbar(im,ax=axs[0],shrink=.7)
    for i,s in enumerate(scores):axs[1].scatter([i]*3,s['seed_validation_half_mse'],color=COL[0],s=20);axs[1].plot(i,s['mean_validation_half_mse'],'_',ms=13,color=COL[3])
    axs[1].set(xticks=range(9),xticklabels=[str(i+1) for i in range(9)],xlabel='协议中的配置序号',ylabel='验证 half-MSE',title='27次运行全部可见');fig.tight_layout();save(fig,'07_search_grid.png')
    contrasts=r['single_factor_contrasts'];fig,ax=plt.subplots(figsize=(9,5.6));labels=[]
    for i,c in enumerate(contrasts):
        ds=c['paired_seed_differences'];ax.scatter(ds,[i]*len(ds),color=COL[0],s=22);ax.plot(c['mean_difference'],i,'|',ms=15,mew=2,color=COL[3]);labels.append(f"{c['changed_parameter']}: {c['from']:g}→{c['to']:g}; {c['fixed_parameter']}={c['fixed_value']:g}")
    ax.axvline(0,color='gray',lw=1);ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='后者 − 前者的验证 half-MSE（负值较好）',title='全部12组相邻取值比较：每点同一初始化种子');ax.invert_yaxis();ax.grid(axis='x',alpha=.2);fig.tight_layout();save(fig,'08_paired_single_factor.png')
    fig,ax=plt.subplots(figsize=(9,4.3));order=np.argsort(test['x']);ax.scatter(test['x'],test['y'],s=9,alpha=.23,color=COL[5],label='独立测试512例')
    for name,label,c in [('baseline','固定参照3模型均值',COL[0]),('selected','验证选出的3模型均值',COL[1]),('least_squares_linear','训练最小二乘直线',COL[3])]:ax.plot(test['x'][order],np.array(final['predictions'][name])[order],label=label,color=c,lw=1.6)
    truth=.8*np.tanh(1.3*test['x'])+.25*np.sin(3*test['x']);ax.plot(test['x'][order],truth[order],color=COL[2],ls=':',label='已知合成条件均值');ax.set(xlabel='输入 x',ylabel='标签或预测',title='测试只作冻结选择后的描述，不再决定下一组参数');ax.legend(fontsize=8,ncol=2);fig.tight_layout();save(fig,'09_final_predictions.png')
    fig,axs=plt.subplots(1,2,figsize=(9,3.9));models=['baseline','selected','least_squares_linear','constant'];labels=['参照','选出','最小二乘','常数']
    for group,off,c in [('left',-.17,COL[0]),('right',.17,COL[1])]:
        vals=[next(v for v in final['ensemble_metrics'] if v['model']==m and v['group']==group)['half_mse'] for m in models];axs[0].bar(np.arange(4)+off,vals,.34,label=group,color=c)
    axs[0].set(xticks=range(4),xticklabels=labels,ylabel='分组 half-MSE',title='左269例，右243例');axs[0].legend()
    residual=np.array(final['predictions']['selected'])-test['y'];axs[1].scatter(test['x'],residual,s=11,alpha=.5,color=COL[2]);axs[1].axhline(0,color='gray',lw=1);axs[1].set(xlabel='x',ylabel='选出模型残差',title='右组噪声更大，不等于自动发现偏差');fig.tight_layout();save(fig,'10_group_errors.png')
    pb=final['paired_bootstrap'];fig,axs=plt.subplots(1,2,figsize=(9,3.8));axs[0].hist(pb['bootstrap_means'],bins=40,color=COL[0],alpha=.85)
    for v in pb['interval_percentile_95']:axs[0].axvline(v,color=COL[3],ls='--')
    axs[0].axvline(0,color='gray');axs[0].set(xlabel='配对平均损失差：选出 − 参照',ylabel='重采样次数',title='2000次按测试ID配对重采样')
    axs[1].scatter(test['x'],pb['row_differences'],s=10,alpha=.45,color=COL[2]);axs[1].axhline(0,color='gray');axs[1].set(xlabel='x',ylabel='同一测试例的损失差',title='平均更好并非每个样本都更好');fig.tight_layout();save(fig,'11_conditional_bootstrap.png')
    fig,axs=plt.subplots(1,2,figsize=(9,3.9));exact=sim['bernoulli_exact'];ks=[v['k'] for v in exact];axs[0].plot(ks,[v['expected_selected_validation_error'] for v in exact],'o-',label='被挑中的验证误差',color=COL[0]);axs[0].axhline(.5,color=COL[1],ls='--',label='独立误差始终0.5');axs[0].set(xlabel='同样好候选个数 K',ylabel='期望误差',xticks=ks,title='精确Bernoulli反例');axs[0].legend(fontsize=8)
    sr=sim['records'];x=np.arange(4);axs[1].plot(x,[v['mean_validation'] for v in sr],'o-',color=COL[0],label='选择所用分数');axs[1].plot(x,[v['mean_audit'] for v in sr],'o-',color=COL[1],label='独立审计分数');axs[1].set(xticks=x,xticklabels=[str(v['candidates']) for v in sr],xlabel='候选数 K',ylabel='2000重复的均值',title='同一真分数1，加独立高斯测量噪声');axs[1].legend(fontsize=8);fig.tight_layout();save(fig,'12_selection_optimism.png')
    fig,axs=plt.subplots(1,2,figsize=(9,3.8))
    for name,label,c in [('reference','参照 η=0.05',COL[0]),('unstable','失稳 η=10',COL[3])]:
        run=next(v for v in runs if v['stage']=='diagnostic' and v['config']['id']==name and v['seed']==4001);pts=run['curve'][:3]
        axs[0].plot([v['step'] for v in pts],[v['gradient_norm'] for v in pts],'o-',label=label,color=c);axs[1].plot([v['step'] for v in pts],[v['proposed_update_ratio'] for v in pts],'o-',label=label,color=c)
    for ax in axs:ax.set(yscale='log',xlabel='当前已接受步数',xticks=[0,1,2]);ax.legend(fontsize=9)
    axs[0].set(ylabel='总目标梯度范数',title='相同初始参数的局部量');axs[1].set(ylabel='拟议 |Δθ| / (|θ| + 1e-12)',title='诊断量没有通用安全阈值');fig.tight_layout();save(fig,'13_update_scale.png')
    print(json.dumps({'figures':13,'all_sources_validated_before_write':True}))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path);parser.add_argument('--destination',type=Path);a=parser.parse_args();main(a.source,a.destination)
