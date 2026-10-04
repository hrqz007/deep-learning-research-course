"""Regenerate all original figures from verified fixed numeric evidence, offline."""
from pathlib import Path
import argparse,hashlib,json,gzip,os,tempfile,shutil
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'dl042-mpl'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(tempfile.gettempdir())/'dl042-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib import font_manager
CJK=Path(os.environ.get('DL_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
if not CJK.is_file():raise RuntimeError('Install Noto Sans CJK or set DL_CJK_FONT to its font file')
font_manager.fontManager.addfont(str(CJK))
FONT=font_manager.FontProperties(fname=str(CJK)).get_name()
ROOT=Path(__file__).resolve().parent
EXPECTED={'data/fixture-hashes.json': 'e3153af9505b63d23a2bdc35916735cca477b4d8cfb43f8a4926650dfaa3b092', 'data/protocol.json': '32062805bbbec66cc2d1dde82a242fb9abab70eaa7b2d6e3062bf51efc8a5420', 'data/test.csv': '14794b91105620fbd6519df4dbb4477e5c39e68e1a39d58fdc8316741bf93292', 'data/train.csv': '6528f8a4eaff59dd33b77292cfe3ce2a0039e90b86d49871990e6556684943e9', 'data/validation.csv': '36ef9b9efdb692ffe6e25df46029cd116c5eb0ce7f33f0e1d830706b6ec6c6eb', 'outputs/final_evaluation.json': '18f0f26550b456ec29c7fa68a52e29ec602f4bb3c8f950160b1177db2b98617f', 'outputs/hand_chain.json': '745cdda30f60114ea2c395fecb5c5441f444336f7a28b36168d28dacfae038a4', 'outputs/identity_leakage.json': '1cfd06997dc74d6fc6116067a55616b4fa8c09c1da3f871053e9a6b1f33f87ff', 'outputs/results.json': '0c26b41bdea78fcaf002d770ce172ae99a10ec24d94554c07293614dd11f449b', 'outputs/training_curves.csv': '89b88dcce9a9cced94a044e1a9924deeabf7d073ce1b28c730e266ed394b5450', 'outputs/training_traces.json.gz': 'd2c9e1439d065d41cb2a56df58daaad7d03365b68cef4262b00929bf653babd1'}
C=['#176A8A','#D97735','#548C52','#855DA6','#B95664']
plt.rcParams.update({'font.family':[FONT,'DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170,'figure.facecolor':'white','axes.facecolor':'#FAFCFE'})
def require(b,s):
 if not b:raise ValueError(s)
def load(name):return json.loads((ROOT/'outputs'/name).read_text())
def main(output=None):
 for name,h in EXPECTED.items():require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,'modified figure input '+name)
 out=Path(output).resolve() if output else ROOT/'figures';require(not out.exists() or out.is_dir(),'output must be directory');require(out not in (ROOT,ROOT/'data',ROOT/'outputs'),'protected directory')
 s=load('results.json');f=load('final_evaluation.json');hand=load('hand_chain.json');leak=load('identity_leakage.json');runs=json.loads(gzip.decompress((ROOT/'outputs/training_traces.json.gz').read_bytes()))['runs']
 import experiment as e
 train=e.load_split('train');validation=e.load_split('validation');test=e.load_split('test')
 with tempfile.TemporaryDirectory(prefix='dl042-figures-') as temp:
  temp=Path(temp)
  def save(name,fig):fig.savefig(temp/name,bbox_inches='tight');plt.close(fig)
  def boxes(ax,labels,positions):
   for text,(x,y) in zip(labels,positions):ax.text(x,y,text,ha='center',va='center',bbox=dict(boxstyle='round,pad=.6',fc='#E7F1F5',ec=C[0]),fontsize=11)
   ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
  fig,ax=plt.subplots(figsize=(10,4));pos=[(.12,.75),(.38,.75),(.65,.75),(.89,.75),(.89,.22),(.65,.22),(.38,.22),(.12,.22)]
  boxes(ax,['任务与抽样单位\n新实体 → 测量','冻结实体拆分\n40 / 32 / 160','训练内预处理\n同一尺度','手算与梯度\n排除实现错','相同神经预算\n12 次拟合','验证选择冻结\n测试不参与','实体配对区间\n保留所有重复','结论与失败\n强基线可能胜出'],pos)
  for a,b in zip(pos,pos[1:]):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color='#7C8895',shrinkA=52,shrinkB=52))
  ax.set_title('完整研究包的审计顺序',pad=15);save('01_audit_chain.png',fig)
  fig,axs=plt.subplots(1,2,figsize=(10,3.8));groups=['A 的第1行','A 的第2行','B 的第1行'];axs[0].bar(np.arange(3)-.18,[1/3]*3,.35,label='每行同权',color=C[0]);axs[0].bar(np.arange(3)+.18,[.25,.25,.5],.35,label='每实体同权',color=C[1]);axs[0].set(xticks=range(3),xticklabels=groups,ylabel='行权重',ylim=(0,.58));axs[0].legend()
  records=s['split_summary'];names=['train','validation','test'];axs[1].bar(np.arange(3)-.18,[records[n]['row_left_mass'] for n in names],.35,color=C[0],label='左区域行比例');axs[1].bar(np.arange(3)+.18,[records[n]['entity_left_mass'] for n in names],.35,color=C[1],label='左区域实体比例');axs[1].set(xticks=range(3),xticklabels=['训练','验证','测试'],ylim=(0,1),ylabel='左区域占有的质量');axs[1].legend();fig.tight_layout();save('02_two_estimands.png',fig)
  fig,axs=plt.subplots(1,3,figsize=(11,3.6),sharex=True,sharey=True)
  for ax,d,title in zip(axs,[train,validation,test],['训练 40 实体','验证 32 实体','测试 160 实体']):
   h=ax.scatter(d['x'][:,0],d['x'][:,1],c=d['y'],s=11,cmap='coolwarm',vmin=-2,vmax=2,alpha=.8);ax.set(title=title,xlabel='x₁');ax.axvline(0,color='#A5AFBA',ls=':',lw=1)
  axs[0].set_ylabel('x₂');fig.colorbar(h,ax=axs.tolist(),label='合成目标 y',shrink=.8);save('03_entity_data.png',fig)
  fig,ax=plt.subplots(figsize=(10,4));pos=[(.07,.5),(.32,.76),(.32,.24),(.62,.5),(.89,.5)];boxes(ax,['x','z₁ = w₁x + b₁\nh₁ = ReLU(z₁)','z₂ = w₂x + b₂\nh₂ = ReLU(z₂)','预测 = a₁h₁ + a₂h₂ + c','qᵢ (预测 − y)² / 2\n上游 = qᵢ rᵢ'],pos)
  for i,j in [(0,1),(0,2),(1,3),(2,3),(3,4)]:ax.add_patch(FancyArrowPatch(pos[i],pos[j],arrowstyle='-|>',mutation_scale=12,color=C[0],shrinkA=30,shrinkB=60))
  ax.text(.5,.04,'三例共享七个参数；正则项 λθ 另加到权重坐标，不加到偏置',ha='center');ax.set_title('纸笔网络的局部计算路径');save('04_hand_graph.png',fig)
  fig,axs=plt.subplots(1,2,figsize=(11,3.8));pnames=hand['parameter_order']
  for k,(ax,step) in enumerate(zip(axs,hand['steps'])):
   M=np.array([[x['float'] for x in r['contribution']] for r in step['rows']]+[[x['float'] for x in step['regularizer_gradient']]])
   im=ax.imshow(M,cmap='coolwarm',vmin=-1,vmax=1,aspect='auto');ax.set(xticks=range(7),xticklabels=pnames,yticks=range(4),yticklabels=['A 第1行','A 第2行','B 第1行','权重惩罚'],title=f'第{k+1}轮的逐路径贡献')
   for i in range(4):
    for j in range(7):ax.text(j,i,f'{M[i,j]:.3f}',ha='center',va='center',fontsize=8,color='#10222D')
  fig.tight_layout();save('05_gradient_ledger.png',fig)
  fig,axs=plt.subplots(1,2,figsize=(10,3.5));ls=[z['data_loss']['float'] for z in hand['steps']]+[hand['third_data_loss']['float']];js=[z['objective']['float'] for z in hand['steps']]+[hand['third_objective']['float']]
  axs[0].plot(range(3),ls,'o-',color=C[0],label='实体数据损失');axs[0].plot(range(3),js,'s-',color=C[1],label='含惩罚目标');axs[0].set(xticks=range(3),xlabel='已经完成的同步更新数',ylabel='目标数值');axs[0].legend()
  states=[[r['prediction']['float'] for r in st['rows']] for st in hand['steps']]+[[r['prediction']['float'] for r in hand['third_forward']]]
  for i,label in enumerate(['A 第1行 y=0','A 第2行 y=1','B 第1行 y=0']):axs[1].plot(range(3),np.array(states)[:,i],'o-',label=label,color=C[i])
  axs[1].set(xticks=range(3),xlabel='已经完成的同步更新数',ylabel='重新前向的预测');axs[1].legend();fig.tight_layout();save('06_hand_updates.png',fig)
  fig,ax=plt.subplots(figsize=(9,3.7));names=['每行同权 MLP','每实体同权 MLP'];y=np.arange(2);ax.barh(y,[2100,2100],color=C[:2]);ax.set(yticks=y,yticklabels=names,xlabel='梯度更新总数',xlim=(0,2550));
  for i in range(2):ax.text(1050,i,'2 个学习率 × 3 种子 × 350 步',ha='center',va='center',color='white');ax.text(2150,i,'25 参数\n470400 行次',va='center')
  ax.text(.5,-.28,'强基线另报 4 次十系数三次多项式拟合；不冒称与梯度下降等计算量',transform=ax.transAxes,ha='center');ax.set_title('相同的是神经训练预算，不能把不同求解器换成同一种成本');save('07_budget_match.png',fig)
  fig,axs=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
  for ax,mode,col in zip(axs,['row','entity'],C):
   for r in runs:
    if r['mode']==mode and r['lr']==s['selected_learning_rates'][mode]:
     ax.plot([v['step'] for v in r['curve']],[v['train_entity_half_mse'] for v in r['curve']],color=col,alpha=.6,lw=1)
     ax.plot([v['step'] for v in r['curve']],[v['validation_entity_half_mse'] for v in r['curve']],color=col,alpha=.6,lw=1,ls='--')
   ax.set(title={'row':'按行训练','entity':'按实体训练'}[mode],xlabel='更新数');ax.text(.95,.93,'实线训练实体风险\n虚线验证实体风险\n每条线对应一个种子',transform=ax.transAxes,ha='right',va='top')
  axs[0].set_ylabel('统一为实体 half-MSE');fig.tight_layout();save('08_training_curves.png',fig)
  fig,ax=plt.subplots(figsize=(8,3.8));x=np.array([.03,.1]);
  for mode,col in zip(['row','entity'],C):
   ss=[v for v in s['network_scores'] if v['mode']==mode];ax.plot(x,[v['mean_validation_entity_half_mse'] for v in ss],'-o',color=col,label={'row':'按行训练','entity':'按实体训练'}[mode])
   for xx,ssr in zip(x,ss):ax.scatter([xx]*3,ssr['seed_losses'],s=16,marker='x',color=col)
  ax.set(xlabel='候选学习率',ylabel='最终验证实体 half-MSE',xticks=x,title='只在各自的两个候选中选择；叉号保留全部三种子');ax.legend();save('09_validation_selection.png',fig)
  fig,ax=plt.subplots(figsize=(9,3.7));b=s['baselines'];c=b['polynomial_candidates'];xx=range(4);vals=[v['validation_entity_half_mse'] for v in c];ax.bar(xx,vals,color=[C[2] if i==b['selected_polynomial_index'] else '#BBCDD2' for i in xx]);ax.set(xticks=list(xx),xticklabels=[str(v['l2']) for v in c],xlabel='三次多项式 ridge λ',ylabel='验证实体 half-MSE',ylim=(0,.06),title='强基线同样用验证集选择，测试不决定 λ');
  for i,vv in enumerate(vals):ax.text(i,vv+.002,f'{vv:.4f}',ha='center')
  save('10_strong_baseline.png',fig)
  fig,ax=plt.subplots(figsize=(10,4));labels=['row','entity','polynomial','linear','constant'];cn=['按行 MLP','按实体 MLP','三次多项式','线性 ridge','常数'];regions=['all','left','right']
  for k,(region,label,col) in enumerate(zip(regions,['全部实体','左区域','右区域'],C)):
   vals=[next(m['entity_half_mse'] for m in f['metrics'] if m['model']==model and m['region']==region) for model in labels];ax.bar(np.arange(5)+(k-1)*.23,vals,.23,label=label,color=col)
  ax.set(xticks=range(5),xticklabels=cn,ylabel='测试实体 half-MSE',title='改进原模型仍未战胜强基线');ax.legend();save('11_test_risks.png',fig)
  fig,axs=plt.subplots(1,2,figsize=(10,3.7));pb=f['primary_paired_bootstrap'];axs[0].hist(pb['differences'],bins=24,color=C[0],alpha=.8);axs[0].axvline(0,color=C[4],ls='--');axs[0].set(xlabel='每实体差值：实体 MLP − 行 MLP',ylabel='测试实体数')
  axs[1].hist(pb['bootstrap_means'],bins=35,color=C[1],alpha=.8);axs[1].axvline(0,color=C[4],ls='--');
  for q in pb['percentile_95']:axs[1].axvline(q,color=C[0],ls=':')
  axs[1].set(xlabel='2000 次整实体重采样的均值',ylabel='重采样次数',title=f"95% 百分位区间 [{pb['percentile_95'][0]:.4f}, {pb['percentile_95'][1]:.4f}]");fig.tight_layout();save('12_entity_bootstrap.png',fig)
  fig,axs=plt.subplots(1,2,figsize=(10,3.7));er=np.array(f['per_entity_half_mse']['entity']);inds=np.argsort(er)[-4:][::-1];
  for k,i in enumerate(inds):
   g=f['test_entity_ids'][i];mask=np.array(test['groups'])==g;axs[0].scatter(test['y'][mask],np.array(f['predictions']['entity'])[mask],color=C[k],label=g,s=36)
  axs[0].plot([-2,2],[-2,2],color='#657586',ls=':');axs[0].set(xlabel='真实标签',ylabel='实体 MLP 预测',title='按实体误差排序展示四个最差实体');axs[0].legend(fontsize=8)
  order=np.argsort(er);axs[1].plot(np.arange(1,161),er[order],color=C[1]);axs[1].set(xlabel='按误差从小到大排列的实体',ylabel='实体 half-MSE',title='完整误差排序，不只展示成功样本');fig.tight_layout();save('13_retained_errors.png',fig)
  fig,axs=plt.subplots(1,2,figsize=(10,3.6));
  for i in range(8):axs[0].plot(range(4),leak['labels'][i],'-o',color=C[i%5],alpha=.75)
  axs[0].set(xticks=range(4),xticklabels=['已见1','已见2','已见3','留出1'],ylabel='身份专属目标',title='同一实体重复观测共享偏移')
  vals=[leak['seen_empirical_half_mse'],leak['unseen_empirical_half_mse']];axs[1].bar([0,1],vals,color=C[:2]);axs[1].scatter([0,1],[leak['seen_theoretical_half_mse'],leak['unseen_theoretical_half_mse']],color='black',marker='_',s=300,label='生成模型的期望');axs[1].set(xticks=[0,1],xticklabels=['已见实体的新行','从未见过的新实体'],ylabel='half-MSE',title='两个部署问题，不能互换评价');axs[1].legend();fig.tight_layout();save('14_identity_leakage.png',fig)
  fig,ax=plt.subplots(figsize=(10,4));ax.axis('off');rows=[('实现证据','全梯度检查、4200 次更新独立重算','支持此实现符合既定公式'),('比较证据','同初始化与预算、冻结选择、整实体区间','支持固定合成数据上的条件性差异'),('仍未证明','换数据、重训练、跨域、总计算优势','需要新协议、新数据和额外实验')]
  for i,(a,b,c) in enumerate(rows):
   yy=.83-i*.31;ax.text(.02,yy,a,color=C[i],weight='bold',fontsize=13);ax.text(.23,yy,b,fontsize=11);ax.text(.23,yy-.11,c,color='#536579',fontsize=10)
  ax.set_title('结论要对应证据的范围',pad=16);save('15_evidence_scope.png',fig)
  files=sorted(temp.glob('*.png'));require(len(files)==15,'figure count')
  out.mkdir(parents=True,exist_ok=True)
  for fpath in files:
   target=out/fpath.name;require(not target.exists() or target.is_file(),'figure target is not file')
  for fpath in files:e.atomic_write(out/fpath.name,fpath.read_bytes())
 return len(files)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);print('figures:',main(p.parse_args().output))
