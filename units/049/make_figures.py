"""Original, data-grounded diagrams; all legends in reserved blank bands."""
from pathlib import Path
import os,argparse,json,gzip
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'../../tmp/049-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle,FancyArrowPatch
import numpy as np
from experiment import ByteBPE,pad_batch
FONT=os.environ.get('DL_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if not Path(FONT).is_file():raise FileNotFoundError('Set DL_CJK_FONT to installed Noto Sans CJK font')
plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'font.size':12,'axes.unicode_minus':False,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
C=['#246a9a','#be692d','#328472'];K=['byte','byte_bpe'];N=['纯字节','字节BPE']
def main(out):
 out.mkdir(parents=True,exist_ok=True);r=json.loads((ROOT/'outputs/results.json').read_text());d=json.loads((ROOT/'data/corpus.json').read_text());audit=r['tokenizer_audit'];h=r['hand'];tr=json.loads(gzip.decompress((ROOT/'outputs/training_traces.json.gz').read_bytes()))
 def save(fig,name):fig.savefig(out/name,bbox_inches='tight');plt.close(fig)
 def legend(fig,ax,n=3):
  handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.02),ncol=n,frameon=False);fig.tight_layout(rect=[0,0,1,.82])
 fig,ax=plt.subplots(figsize=(10,3));ax.axis('off');labels=['文字：材料A\n3个Unicode码点','UTF-8：7字节\ne6 9d 90 e6 96 99 41','字节ID：每字节+4\n外加BOS / EOS','查表 E[id]\n每个位置得到D维向量']
 for j,label in enumerate(labels):
  x=.01+j*.25;ax.add_patch(Rectangle((x,.36),.22,.46,facecolor='#e8f0f7',edgecolor='#54728a'));ax.text(x+.11,.59,label,ha='center',va='center',fontsize=11)
  if j<3:ax.add_patch(FancyArrowPatch((x+.223,.59),(x+.247,.59),arrowstyle='-|>',mutation_scale=12))
 ax.text(.5,.11,'本讲不做Unicode规范化；16次BPE合并在字节ID之后、边界符之前应用',ha='center');save(fig,'01_pipeline.png')
 fig,ax=plt.subplots(figsize=(10,3.7));x=np.arange(7);labels=['材料A','café\nU+00E9','cafe +\nU+0301','U+1F469\nZWJ U+1F52C','字面量\n<PAD>','空串','new +\n希腊字母/emoji']
 for j,(key,label) in enumerate([('unicode_codepoints','码点数'),('utf8_bytes','UTF-8字节数'),('bpe_content_tokens','本BPE内容token数')]):ax.bar(x+(j-1)*.23,[p[key] for p in audit['probes']],.23,label=label,color=C[j])
 ax.set_xticks(x,labels);ax.set_ylabel('不含控制符的计数');ax.set_ylim(0,12);legend(fig,ax);save(fig,'02_counts.png')
 hist=audit['bpe_history'];initial=sum(len(t.encode()) for t in d['train']);fig,ax=plt.subplots(1,2,figsize=(10,3.5));ax[0].plot(range(17),[initial]+[v['content_tokens_after'] for v in hist],'o-',color=C[0]);ax[0].set(xlabel='已执行合并次数',ylabel='48篇训练文档的内容token总数',xticks=[0,4,8,12,16]);ax[0].set_title('词表增加16项，内容token减少408个');ax[1].bar(range(1,17),[v['frequency_count_including_overlap'] for v in hist],color=C[1]);ax[1].set(xlabel='合并序号',ylabel='选中相邻对出现次数（可重叠）',xticks=[1,4,8,12,16]);ax[1].set_title('频次不总等于实际替换数');fig.tight_layout();save(fig,'03_bpe_history.png')
 fig,ax=plt.subplots(figsize=(10,3.3));ax.axis('off');rows=[('文档一','a a a a','频次3：相邻位置(0,1)、(1,2)、(2,3)','左到右非重叠替换 → [aa] [aa]'),('文档二/三','[a] | [b]','文档边界不参与相邻对计数','不能凭跨文档的相邻顺序生成 [ab]')]
 for j,(title,text,note,outcome) in enumerate(rows):
  y=.82-j*.46;ax.text(.02,y,title,fontweight='bold');ax.text(.23,y,text,color=C[j],fontsize=15);ax.text(.02,y-.14,note);ax.text(.02,y-.28,outcome,color='#334155')
 save(fig,'04_merge_boundaries.png')
 fig,ax=plt.subplots(1,2,figsize=(10,3.8));E=np.array([0,.2,.4,.3,.5,-.25]);ids=[4,4,5];M=np.eye(6)[ids];im=ax[0].imshow(M,cmap='Blues',vmin=0,vmax=1);ax[0].set(xticks=range(6),xticklabels=['PAD','BOS','EOS','UNK','A','B'],yticks=range(3),yticklabels=['位置0:A','位置1:A','位置2:B']);ax[0].set_title('one-hot选择矩阵：每行只选一个ID')
 for i in range(3):
  for j in range(6):ax[0].text(j,i,str(int(M[i,j])),ha='center',va='center',color='white' if M[i,j] else '#333')
 ax[1].bar(['A第一次','A第二次','B第一次'],E[ids],color=[C[0],C[0],C[1]]);ax[1].axhline(0,color='#777',lw=.6);ax[1].set_title('one-hot × E = E[ids]；重复读同一行');ax[1].set_ylabel('D=1时的嵌入值');fig.tight_layout();save(fig,'05_lookup.png')
 fig,ax=plt.subplots(1,3,figsize=(11,3));x,y,mask,_=pad_batch([[1,4,4,5,2],[1,5,2],[1,2]],6,pad_to=5)
 for a,v,title in zip(ax,[x,y,mask.astype(int)],['输入x：去掉末尾EOS','目标y：去掉开头BOS','mask：只计有效目标']):
  a.imshow(v,cmap='Blues',vmin=0,vmax=5 if title[0]!='m' else 1);a.set_title(title);a.set(xticks=range(5),yticks=range(3),yticklabels=['A A B','B','空串'])
  for ij in np.ndindex(v.shape):a.text(ij[1],ij[0],str(v[ij]),ha='center',va='center',color='white' if v[ij]>(2.5 if title[0]!='m' else .5) else '#333')
 fig.suptitle('0=PAD，1=BOS，2=EOS，4=A，5=B；空串仍有BOS→EOS一个目标',y=1.07);fig.tight_layout();save(fig,'06_padding.png')
 fig,ax=plt.subplots(figsize=(10,3.4));values=np.array([[v['float'] for v in s['contribution']][4:] for s in h[0]['samples']]+[[v['float'] for v in h[0]['gradient']][4:]]);im=ax.imshow(values,cmap='RdBu_r',vmin=-.55,vmax=.55,aspect='auto');ax.set(xticks=range(4),xticklabels=['E[A]','E[B]','w','b'],yticks=range(3),yticklabels=['样本1（含1/2）','样本2（含1/2）','共享参数总梯度']);ax.set_title('同ID在时间轴/批次轴上累加；第一次更新的w梯度恰好抵消')
 for ij in np.ndindex(values.shape):ax.text(ij[1],ij[0],f'{values[ij]:.6f}',ha='center',va='center',color='white' if abs(values[ij])>.35 else '#222')
 fig.tight_layout();save(fig,'07_scatter.png')
 fig,ax=plt.subplots(1,2,figsize=(10,3.4))
 for i in range(2):ax[0].plot(range(3),[v['samples'][i]['loss']['float'] for v in h],'o-',label=f'样本{i+1}',color=C[i])
 ax[0].plot(range(3),[v['loss']['float'] for v in h],'s--',label='平均',color=C[2]);ax[0].set(xlabel='同步更新次数',ylabel='half-MSE',xticks=[0,1,2]);ax[0].set_title('每次更新后从查表重新前向')
 for i,label in [(4,'E[A]'),(5,'E[B]')]:ax[1].plot(range(3),[v['theta'][i]['float'] for v in h],'o-',label=label)
 ax[1].set(xlabel='同步更新次数',ylabel='嵌入行值',xticks=[0,1,2]);ax[1].set_title('E[A]与E[B]都由两个位置/样本路径影响');handles,labels=ax[0].get_legend_handles_labels();hh,ll=ax[1].get_legend_handles_labels();fig.legend(handles+hh,labels+ll,loc='upper center',bbox_to_anchor=(.5,1.02),ncol=5,frameon=False);fig.tight_layout(rect=[0,0,1,.82]);save(fig,'08_hand_updates.png')
 fig,ax=plt.subplots(1,2,figsize=(10,3.4))
 for a,kind,label in zip(ax,K,N):
  for t in tr:
   if t['kind']==kind:a.plot([s['step'] for s in t['states']],[s['train_nll_per_target'] for s in t['states']],label=str(t['seed']),lw=1.3)
  a.set_title(label+'：全部3个种子');a.set(xlabel='全批次SGD更新次数',ylabel='本词表内每有效token训练NLL')
 legend(fig,ax[0]);save(fig,'09_training.png')
 fig,ax=plt.subplots(1,2,figsize=(10,3.8));x=np.arange(2)
 for j,metric in enumerate(['mean_nll_per_target','bits_per_utf8_byte']):
  vals=[r['summary'][k][metric] for k in K];base=[r['unigram_baselines'][k]['metrics']['test'][metric] for k in K];ax[j].bar(x-.18,vals,.35,label='嵌入+线性头（3种子均值）',color=C[0]);ax[j].bar(x+.18,base,.35,label='加一平滑unigram',color=C[1]);
  for i,k in enumerate(K):ax[j].scatter([i-.18]*3,[v['metrics']['test'][metric] for v in r['runs'] if v['kind']==k],color='black',s=13)
  ax[j].set_xticks(x,N);ax[j].set_ylabel('nats / target token' if j==0 else 'bits / UTF-8 content byte');ax[j].set_title('左侧单位随词表变化' if j==0 else '右侧统一字节单位，但不是全文概率边缘化')
 legend(fig,ax[0],2);save(fig,'10_metrics.png')
 fig,ax=plt.subplots(1,2,figsize=(10,3.6));byte=ByteBPE();bpe=ByteBPE(audit['bpe']['merges']);lengths=[[len(t.encode(text))-1 for text in d['train']] for t in [byte,bpe]]
 for v,label,c in zip(lengths,N,C):ax[0].plot(range(48),v,'o-',label=label,color=c,ms=3,lw=.7)
 ax[0].set(xlabel='固定训练文档编号',ylabel='有效目标数（含EOS）');ax[0].set_title('同一文档压缩程度不同，token数不能当字数')
 params=[next(v for v in r['runs'] if v['kind']==k)['parameter_coordinates'] for k in K];ax[1].bar(N,params,color=C[:2]);ax[1].set_ylabel('参数坐标数（包括固定PAD的8维）');ax[1].set_title('V增大使E与输出头一起变大')
 for i,n in enumerate(params):ax[1].text(i,n+50,str(n),ha='center')
 ax[1].set_ylim(0,5300);legend(fig,ax[0],2);save(fig,'11_budgets.png');print(json.dumps({'figures':11,'font':FONT},ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'figures');main(p.parse_args().output)
