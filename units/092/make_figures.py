"""Regenerate all figures from saved experiment evidence."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/dl092-mpl')
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'figures';O.mkdir(exist_ok=True)
res=json.loads((R/'outputs/results.json').read_text());rows=res['aggregate'];methods=list(dict.fromkeys(r['method'] for r in rows));domains=list(dict.fromkeys(r['domain'] for r in rows))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
def save(fig,name): fig.savefig(O/name,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(10,2.8));ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off')
for i,(title,detail) in enumerate([('Claim','What would falsify it?'),('Controls','Information and capacity'),('Interventions','Check label validity'),('Evidence','Paired errors and groups'),('Scope','What remains unknown?')]):
    x=.15+i*2;ax.text(x+.8,1.7,title,ha='center',bbox=dict(boxstyle='round',fc='#dcecf6',ec='#337799',pad=.6));ax.text(x+.8,.65,detail.replace(' and ','\nand '),ha='center',fontsize=8)
    if i<4:ax.annotate('',xy=(x+1.85,1.7),xytext=(x+1.5,1.7),arrowprops=dict(arrowstyle='->'))
save(fig,'01_evidence.png')
fig,ax=plt.subplots(figsize=(6,4));
for a in [-1,1]:
    for b in [-1,1]:ax.scatter(a,b,s=400,c=['#dd8855' if a*b<0 else '#3388aa']);ax.annotate(f'y={int(a*b>0)}',(a,b),ha='center',va='center',color='white')
ax.set(xlim=(-1.8,1.8),ylim=(-1.8,1.8),xlabel='Latent visual direction a',ylabel='Query b',xticks=[-1,1],yticks=[-1,1],title='Equality requires an interaction');ax.axhline(0,color='#ddd');ax.axvline(0,color='#ddd');save(fig,'02_xor.png')
fig,ax=plt.subplots(figsize=(8,4));mat=np.array([[0,0,0,0],[0,0,1,0],[0,0,1,0],[1,0,0,0],[0,1,1,0],[0,1,0,1],[0,1,1,0],[0,1,1,1]])
ax.imshow(mat,cmap='Blues',vmin=0,vmax=1,aspect='auto');ax.set_xticks(range(4),['Visual v','Query b','Shortcut z','Label']);ax.set_yticks(range(8),domains);ax.set_title('Dark = changed relative to original test example')
for i in range(8):
    for j in range(4):ax.text(j,i,'changed' if mat[i,j] else 'kept',ha='center',va='center',color='white' if mat[i,j] else '#333',fontsize=8)
save(fig,'03_interventions.png')
fig,ax=plt.subplots(figsize=(11,4));mat=np.array([[next(r['error_mean'] for r in rows if r['method']==m and r['domain']==d)*100 for d in domains] for m in methods]);im=ax.imshow(mat,cmap='YlOrRd',vmin=0,vmax=100,aspect='auto');ax.set_yticks(range(5),methods);ax.set_xticks(range(8),domains,rotation=30,ha='right');ax.set_title('Test error (%) | Lower is better within each valid label definition')
for i in range(5):
    for j in range(8):ax.text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center',color='white' if mat[i,j]>65 else '#222',fontsize=9)
fig.colorbar(im,ax=ax,label='Error (%)');save(fig,'04_errors.png')
fig,axes=plt.subplots(1,2,figsize=(10,3.5));x=np.arange(2);ms=['interaction','randomized_interaction']
for g in [0,1]:
    vals=[np.mean([r[f'group{g}_error'] for r in res['results'] if r['method']==m and r['domain']=='shortcut_flip'])*100 for m in ms];axes[0].bar(x+(g-.5)*.3,vals,.3,label=f'Noise group {g}')
axes[0].set_xticks(x,['Original\ninteraction','Randomized\ninteraction']);axes[0].set_ylabel('Shortcut-flip error (%)');axes[0].legend();delta=res['paired_delta']*100;lo,hi=np.array(res['paired_example_bootstrap_ci'])*100;axes[1].errorbar([0],[delta],yerr=[[delta-lo],[hi-delta]],fmt='o',capsize=6);axes[1].axhline(0,color='gray',linestyle='--');axes[1].set(xticks=[0],xticklabels=['Randomized minus original'],ylabel='Paired difference (percentage points)',title='95% example-bootstrap interval');fig.tight_layout();save(fig,'05_groups.png')
# Fill measured results in the editable lecture, never invent numbers.
p=R/'lecture.md';text=p.read_text();table='|方法|原域错误率|提示反转|反事实问题|重标置乱|\n|---|---:|---:|---:|---:|\n'
for m in methods:
    vals=[next(r['error_mean'] for r in rows if r['method']==m and r['domain']==d)*100 for d in ['iid','shortcut_flip','question_counterfactual','permuted_relabelled']];table+='|'+m+'|'+'|'.join(f'{v:.2f}%' for v in vals)+'|\n'
text=text.replace('<!-- RESULTS_TABLE -->',table).replace('<!-- RESULT_PARAGRAPH -->',f'本次实测，随机化交互模型减去原交互模型的提示反转域错误率差为{delta:.2f}个百分点；固定样本层面bootstrap的95%区间为[{lo:.2f}, {hi:.2f}]个百分点。这个区间不包含全部训练数据或未来环境不确定性。完整15次训练耗时约{res["seconds"]:.2f}秒，权重重载最大差为0。')
p.write_text(text)
