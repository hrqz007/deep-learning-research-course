"""Original finite-precision teaching figures; all measurements from local outputs."""
from pathlib import Path
import json,csv,math,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/dl014-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle,FancyArrowPatch
BASE=Path(__file__).resolve().parent;OUT=BASE/'figures';OUT.mkdir(exist_ok=True)
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';fm.fontManager.addfont(font)
plt.rcParams.update({'font.family':fm.FontProperties(fname=font).get_name(),'font.size':11,'axes.unicode_minus':False})
blue='#215e9c';orange='#b64f16';green='#287b58'
r=json.loads((BASE/'outputs/results.json').read_text())
def save(fig,name):fig.savefig(OUT/name,dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)
fig,ax=plt.subplots(figsize=(8.5,2.8));v=[1,1.25,1.5,1.75,2,2.5,3,3.5,4,5,6,7]
ax.hlines(0,.8,7.2,color='#8899aa');ax.scatter(v,[0]*len(v),color=blue,s=55)
for x in v:ax.text(x,-.12,f'{x:g}',ha='center',fontsize=9)
ax.set(xlim=(.8,7.2),ylim=(-.4,.65));ax.axis('off')
for a,b,lab in [(1,2,'间隔 0.25'),(2,4,'间隔 0.5'),(4,7,'间隔 1')]:
 ax.annotate('',xy=(a,.35),xytext=(b,.35),arrowprops={'arrowstyle':'<->','color':orange});ax.text((a+b)/2,.46,lab,ha='center',color=orange)
ax.set_title('玩具格式 (1.b1 b2) × 2^e，仅3个有效二进制位');save(fig,'01_spacing.png')
fig,axs=plt.subplots(1,2,figsize=(8.5,3.1),sharey=True)
for ax,dt in zip(axs,['float32','float64']):
 vals=r['cast_information_loss'][dt];ax.bar(['分数1','分数2'],vals,color=[blue,orange]);ax.set(ylim=(0,1),title=dt)
 for i,x in enumerate(vals):ax.text(i,x+.025,f'{x:.6f}',ha='center')
 ax.set_xlabel('存值 '+('100000000, 100000000' if dt=='float32' else '100000000, 100000001'),fontsize=9)
axs[0].set_ylabel('稳定softmax权重');fig.suptitle('原始分数 (100000000,100000001)');fig.tight_layout();save(fig,'02_cast.png')
fig,ax=plt.subplots(figsize=(8.5,2.9))
for y,dt,col in [(0,'float32',blue),(1,'float64',orange)]:
 f=r['formats'][dt];a,b,c=[math.log10(f[k]) for k in ['smallest_subnormal','smallest_normal','max']]
 ax.plot([a,b],[y,y],'--',color=col,lw=4);ax.plot([b,c],[y,y],color=col,lw=4);ax.scatter([a,b,c],[y]*3,color=col,zorder=5)
 ax.text(a,y+.13,f'次正规起点\n{a:.1f}',ha='center',fontsize=9);ax.text(b,y-.3,f'正规起点\n{b:.1f}',ha='center',fontsize=9);ax.text(c,y+.13,f'最大值\n{c:.1f}',ha='center',fontsize=9)
ax.set(yticks=[0,1],yticklabels=['float32','float64'],xlabel='log10(绝对值)',xlim=(-375,355),ylim=(-.48,1.5));ax.grid(axis='x',alpha=.2);fig.tight_layout();save(fig,'03_range.png')
fig,axs=plt.subplots(1,2,figsize=(8.5,3.3),sharey=True)
for ax,dt in zip(axs,['float32','float64']):
 rows=r['cancellation'][dt];xs=[row['stored_x'] for row in rows]
 for key,col in [('direct',orange),('rationalized',blue)]:
  errors=[abs(row[key]-row['reference'])/abs(row['reference']) for row in rows];ax.loglog(xs,np.maximum(errors,1e-18),'o-',label='直接相减' if key=='direct' else '有理化',color=col)
 ax.set(title=dt,xlabel='实际存储的 x');ax.grid(alpha=.2);ax.legend(fontsize=9)
axs[0].set_ylabel('相对误差（显示下限 1e-18）');fig.tight_layout();save(fig,'04_cancellation.png')
fig,ax=plt.subplots(figsize=(9,3.3));ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
def box(x,y,w,h,t,col):ax.add_patch(Rectangle((x,y),w,h,edgecolor=col,facecolor='#f1f5f8',lw=1.5));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=10,color=col)
def arrow(a,b,col):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color=col))
box(.1,2.25,2.5,1,'(1000,1001,999)',orange);box(3.7,2.25,2.1,1,'exp → inf\n求和 → inf',orange);box(7,2.25,2.8,1,'inf / inf → NaN',orange);arrow((2.6,2.75),(3.7,2.75),orange);arrow((5.8,2.75),(7,2.75),orange)
box(.1,.3,2.5,1,'减最大值1001\n(-1,0,-2)',blue);box(3.7,.3,2.1,1,'exp值 ≤ 1\nS ≈ 1.503215',blue);box(7,.3,2.8,1,'p ≈ (0.244728,\n0.665241,0.090031)',blue);arrow((2.6,.8),(3.7,.8),blue);arrow((5.8,.8),(7,.8),blue);save(fig,'05_softmax.png')
rows=list(csv.DictReader((BASE/'outputs/difference_scan.csv').open()));fig,ax=plt.subplots(figsize=(8.5,3.7))
for dt,col in [('float32',orange),('float64',blue)]:
 good=[r for r in rows if r['dtype']==dt and r['status']=='computed'];bad=[r for r in rows if r['dtype']==dt and r['status']!='computed'];ax.loglog([float(r['stored_h']) for r in good],[max(float(r['absolute_error']),1e-18) for r in good],'o-',color=col,label=f'{dt}，另有{len(bad)}个拒绝步长')
ax.set(xlabel='dtype中实际存储的差分 h',ylabel='对实际存储 x 的导数绝对误差',title='exp中心差分：先减少截断误差，再受舍入影响');ax.grid(alpha=.2);ax.legend();fig.tight_layout();save(fig,'06_difference.png')
