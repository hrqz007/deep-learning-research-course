"""Original gate-flow diagrams and measured fixed-protocol results."""
from pathlib import Path
import os,json,gzip,argparse
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'../../tmp/051-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle,FancyArrowPatch
import numpy as np
FONT=os.environ.get('DL_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if not Path(FONT).is_file():raise FileNotFoundError('Set DL_CJK_FONT to installed Noto Sans CJK font')
plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'font.size':12,'axes.unicode_minus':False,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
C=['#246a9a','#bd6b2d','#328472'];K=['rnn','gru'];LABEL=['tanh RNN','GRU']
def main(out):
 out.mkdir(parents=True,exist_ok=True);r=json.loads((ROOT/'outputs/results.json').read_text());h=json.loads((ROOT/'outputs/hand_ledger.json').read_text())['states'];data=json.loads((ROOT/'data/sequences.json').read_text());tr=json.loads(gzip.decompress((ROOT/'outputs/training_traces.json.gz').read_bytes()))
 def save(fig,name):fig.savefig(out/name,bbox_inches='tight');plt.close(fig)
 def legend(fig,ax,n=3):fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.02),ncol=n,frameon=False);fig.tight_layout(rect=[0,0,1,.82])
 def box(ax,x,y,w,hh,text):ax.add_patch(Rectangle((x,y),w,hh,facecolor='#e8f0f7',edgecolor='#567188'));ax.text(x+w/2,y+hh/2,text,ha='center',va='center',fontsize=11)
 def arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color='#4a6073'))
 fig,ax=plt.subplots(figsize=(11,3.7));ax.axis('off');box(ax,.02,.36,.15,.23,'x 与旧 h');box(ax,.25,.66,.19,.2,'r = sigmoid\n重置候选分支');box(ax,.25,.12,.19,.2,'z = sigmoid\n保留旧状态比例');box(ax,.55,.55,.20,.25,'n = tanh\n输入仿射\n+ r × 隐状态仿射');box(ax,.79,.25,.20,.24,'h新 = (1-z)n\n+ z h旧');arrow(ax,(.17,.52),(.25,.74));arrow(ax,(.17,.43),(.25,.22));arrow(ax,(.44,.75),(.55,.68));arrow(ax,(.44,.22),(.79,.36));arrow(ax,(.75,.62),(.85,.49));ax.text(.5,.01,'reset-after约定：r乘在U_n h + b_hn整体上；图中公式明确保留旧h的直路',ha='center',fontsize=11);save(fig,'01_gru_flow.png')
 fig,ax=plt.subplots(figsize=(10,3.4));ax.axis('off');box(ax,.02,.57,.15,.2,'旧c=1.2');box(ax,.26,.57,.18,.2,'× f=0.8');box(ax,.26,.18,.18,.2,'i×g=.25×(-.5)');box(ax,.57,.48,.18,.2,'相加：c=.835');box(ax,.79,.48,.19,.2,'o×tanh(c)\no=.6');arrow(ax,(.17,.67),(.26,.67));arrow(ax,(.44,.67),(.57,.58));arrow(ax,(.44,.28),(.64,.48));arrow(ax,(.75,.58),(.79,.58));ax.text(.5,.04,'LSTM分开保存c与h；这里固定门值用于解析演示，实际门也依赖x与旧h',ha='center');save(fig,'02_lstm_flow.png')
 fig,ax=plt.subplots(figsize=(10,3.4));T=np.arange(101)
 for f,c in zip([.5,.9,.99],C):ax.semilogy(T,f**T,label=f'固定门={f}',color=c)
 ax.set(xlabel='经过时间步数',ylabel='只沿加性记忆直路的导数乘积',title='解析探针：f^T 或 z^T；不是完整循环Jacobian');legend(fig,ax);save(fig,'03_memory_product.png')
 steps=[h[0]['samples'][0]['steps'][0],h[0]['samples'][0]['steps'][1],h[0]['samples'][1]['steps'][0]];v=np.array([[s[k] for k in ['r','z','n','h']] for s in steps]);fig,ax=plt.subplots(figsize=(10,3.3));ax.imshow(v,cmap='RdBu_r',vmin=-.62,vmax=.62,aspect='auto');ax.set(xticks=range(4),xticklabels=['重置门r','保留门z','候选n','新状态h'],yticks=range(3),yticklabels=['样本1 t1','样本1 t2','样本2 t1']);ax.set_title('初始同一组参数，两条独立序列均从h0=0开始')
 for ij in np.ndindex(v.shape):ax.text(ij[1],ij[0],f'{v[ij]:.6f}',ha='center',va='center',color='white' if abs(v[ij])>.4 else '#222')
 fig.tight_layout();save(fig,'04_hand_gates.png')
 back=h[0]['samples'][0]['backward'];fig,ax=plt.subplots(figsize=(10,3.3));x=np.arange(2)
 for j,(k,label,c) in enumerate([('dh_prev_direct','沿z直路',C[0]),('dh_prev_via_gates','经候选/门控路径',C[1]),('dh_prev_total','总和',C[2])]):ax.bar(x+(j-1)*.24,[s[k] for s in back],.24,label=label,color=c)
 ax.axhline(0,color='#666',lw=.7);ax.set_xticks(x,['t2传回h1','t1传回h0']);ax.set_ylabel('对旧状态的上游梯度');ax.set_title('完整BPTT要相加所有路径；不能只保留z×dh');legend(fig,ax);save(fig,'05_hand_paths.png')
 v=np.array([s['gradient_contribution'] for s in h[0]['samples']]+[h[0]['gradient']]);fig,axes=plt.subplots(2,1,figsize=(10,5.4))
 for part,ax in enumerate(axes):
  first=part*7;partv=v[:,first:first+7];ax.imshow(partv,cmap='RdBu_r',vmin=-.18,vmax=.18,aspect='auto');ax.set(xticks=range(7),xticklabels=h[0]['parameter_order'][first:first+7],yticks=range(3),yticklabels=['样本1','样本2','总梯度']);ax.tick_params(axis='x',labelsize=12)
  for ij in np.ndindex(partv.shape):ax.text(ij[1],ij[0],f'{partv[ij]:.4f}',ha='center',va='center',fontsize=12,color='white' if abs(partv[ij])>.12 else '#222')
 fig.suptitle('14个参数分两行展示：样本内先累加时间，样本间再相加；已含批平均',fontsize=12);fig.tight_layout();save(fig,'06_parameter_sums.png')
 fig,ax=plt.subplots(figsize=(10,3.3))
 for i in range(2):ax.plot(range(3),[s['samples'][i]['individual_half_mse'] for s in h],'o-',label=f'样本{i+1}',color=C[i])
 ax.plot(range(3),[s['loss'] for s in h],'s--',label='平均',color=C[2]);ax.set(xlabel='同步SGD更新次数，学习率0.2',ylabel='half-MSE',xticks=[0,1,2],title='总损失下降，样本2反而变差：共享目标不是逐样本保证');legend(fig,ax);save(fig,'07_hand_updates.png')
 fig,ax=plt.subplots(figsize=(11,3.7));ax.axis('off');box(ax,.01,.6,.23,.2,'编码：BOS A B C EOS');box(ax,.32,.6,.24,.2,'仅传最终h\n固定12维上下文');box(ax,.65,.6,.33,.2,'解码：BOS→C→B→A→EOS');arrow(ax,(.24,.7),(.32,.7));arrow(ax,(.56,.7),(.65,.7));box(ax,.04,.2,.4,.23,'训练：下一输入来自真实前缀\n目标右移，EOS参与loss');box(ax,.55,.2,.4,.23,'推理：下一输入来自自己输出\n首个EOS停止，最多10步');ax.text(.5,.03,'独立文档分别h0=0；补齐处冻结状态；encoder和decoder参数不同，嵌入表共享',ha='center',fontsize=11);save(fig,'08_seq2seq.png')
 fig,ax=plt.subplots(1,3,figsize=(12,3.7));keys=['teacher_forced_token_accuracy','free_aligned_token_accuracy','free_sequence_exact'];titles=['真实前缀下token准确率','自由生成按金标位置计分','整条序列完全相同（含EOS）']
 for a,key,title in zip(ax,keys,titles):
  a.bar(LABEL,[r['summary'][k]['test'][key] for k in K],color=C[:2]);a.set(ylim=(0,1),ylabel='比例',title=title)
  for i,k in enumerate(K):a.scatter([i]*3,[v['metrics']['test'][key] for v in r['runs'] if v['kind']==k],color='black',s=16)
 fig.suptitle('同长度范围测试集：柱为3种子均值，点为完整种子；token准确率不是序列成功率',y=1.06,fontsize=11);fig.tight_layout();save(fig,'09_teacher_free.png')
 fig,ax=plt.subplots(1,2,figsize=(11,3.6))
 for j,kind in enumerate(K):
  for t in tr:
   if t['kind']==kind:ax[j].plot([s['step'] for s in t['states']],[s['train_nll'] for s in t['states']],label=str(t['seed']),lw=1)
  ax[j].set(title=LABEL[j],xlabel='全批次SGD更新次数',ylabel='训练有效token NLL')
 legend(fig,ax[0]);save(fig,'10_traces.png')
 def fmt(seq):return ' '.join({0:'PAD',1:'BOS',2:'EOS',3:'UNK',4:'A',5:'B',6:'C',7:'D'}[v] for v in seq)
 rows=[]
 for i in range(5):rows.append([str(i),fmt(data['test'][i]),fmt(list(reversed(data['test'][i]))+[2])]+[fmt(next(v for v in r['runs'] if v['kind']==k and v['seed']==5111)['metrics']['test']['decoded'][i]) for k in K])
 fig,ax=plt.subplots(figsize=(11,3.8));ax.axis('off');table=ax.table(cellText=rows,colLabels=['编号','输入内容','正确目标','RNN seed5111','GRU seed5111'],colWidths=[.075,.175,.24,.255,.255],cellLoc='center',loc='center');table.auto_set_font_size(False);table.set_fontsize(14);table.scale(1,2.25);ax.set_title('测试集最前5项，不按成功或失败筛选；完整64项×6模型保留',pad=20);save(fig,'11_decoded_examples.png')
 fig,ax=plt.subplots(1,2,figsize=(11,3.7))
 for kind,label,c in zip(K,LABEL,C):
  values=[];counts=[]
  for L in range(2,7):
   split='test' if L<=4 else 'long_test';idx=[i for i,s in enumerate(data[split]) if len(s)==L];counts.append(len(idx));values.append(float(np.mean([run['metrics'][split]['correct'][i] for run in r['runs'] if run['kind']==kind for i in idx])))
  ax[0].plot(range(2,7),values,'o-',label=label,color=c)
 ax[0].set(xlabel='源序列内容长度',ylabel='自由生成整序列准确率',ylim=(-.03,1.03),xticks=range(2,7),title='长度2..4同范围测试；5..6为外推测试')
 for i,kind in enumerate(K):ax[1].bar(np.array([0,1])+(i-.5)*.3,[r['summary'][kind][split]['teacher_forced_nll'] for split in ['test','long_test']],.3,color=C[i],label=LABEL[i])
 ax[1].set(xticks=[0,1],xticklabels=['同长度范围','更长序列'],ylabel='真实前缀NLL',title='长序列exact全0，两模型NLL仍不同');legend(fig,ax[0],2);save(fig,'12_length_generalization.png');print(json.dumps({'figures':12,'test_counts_by_length2to6':counts}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'figures');main(p.parse_args().output)
