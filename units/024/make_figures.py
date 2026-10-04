"""Original fixed figures. Validate every teaching input/output before plotting."""
from experiment import require_teaching_inputs
require_teaching_inputs()
from pathlib import Path
import json,csv,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from experiment import sigmoid,binary_loss_gradient,multiclass_loss_gradient,log_softmax,objective,read_data
HERE=Path(__file__).resolve().parent;OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
f=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if f.exists():font_manager.fontManager.addfont(str(f));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(f)).get_name()
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'axes.unicode_minus':False})
C=['#287f9e','#da923c','#aa4f74','#479f7c'];result=json.loads((HERE/'outputs/results.json').read_text())
def save(n):plt.tight_layout();plt.savefig(OUT/n,bbox_inches='tight',facecolor='white');plt.close()
def readcsv(n):
 with (HERE/'outputs'/n).open() as f:return list(csv.DictReader(f))

zs=np.linspace(-7,7,201);fig,ax=plt.subplots(1,2,figsize=(9,3.4));ax[0].plot(zs,sigmoid(zs),color=C[0]);ax[0].axhline(.5,color='#666',ls=':');ax[0].axvline(0,color='#666',ls=':');ax[0].set(xlabel='logit z',ylabel='p = P(Y=1|x)',title='连续概率，不是硬标签')
for y,col in [(0,C[1]),(1,C[2])]:ax[1].plot(zs,[binary_loss_gradient([z],[y])[0] for z in zs],color=col,label=f'y={y}')
ax[1].set(xlabel='logit z',ylabel='单例BCE',title='惩罚高置信但错误的预测');ax[1].legend();save('01_sigmoid_loss.png')

fig,ax=plt.subplots(1,2,figsize=(9.5,3.6));z=np.arange(0,46,dtype=float);stable=-np.array([binary_loss_gradient([v],[1])[1][0] for v in z]);naive=1-sigmoid(z)
ax[0].semilogy(z,stable,color=C[0],label='稳定 |梯度|');mask=naive>0;ax[0].semilogy(z[mask],naive[mask],'.',color=C[1],label='直接1−sigmoid(z)，仅画非零');ax[0].annotate('z=37后直接式为0',xy=(37,stable[37]),xytext=(14,1e-18),arrowprops={'arrowstyle':'->'},fontsize=9);ax[0].set(xlabel='正确正例的z',ylabel='梯度绝对值（对数轴）');ax[0].legend(fontsize=8)
ax[1].axis('off');tab=ax[1].table(cellText=[['−1000','1','1000','−1'],['1000','0','1000','1'],['1000','1','0*','0*']],colLabels=['z','y','稳定BCE','梯度'],cellLoc='center',bbox=[.03,.35,.94,.55]);tab.auto_set_font_size(False);tab.set_fontsize(11);ax[1].text(.5,.15,'* 此处0来自float64下溢\n不是精确实数结论',ha='center',fontsize=11,color=C[2]);save('02_extreme_binary.png')

fig,ax=plt.subplots(figsize=(8,4));verts=np.array([[0,0],[1,0],[.5,math.sqrt(3)/2]]);tri=np.vstack([verts,verts[0]]);ax.plot(tri[:,0],tri[:,1],color='#333')
for i,(x,y) in enumerate(verts):ax.text(x,y-.055 if i<2 else y+.035,f'p{i}=1',ha='center',fontsize=13)
pts=[]
label_positions={0:(.55,.30),1:(.22,.30),2:(.20,.13),4:(.06,.11),6:(.17,-.075)}
for t in [0,1,2,4,6]:
 p=np.exp(log_softmax([[t,0,0]]))[0];point=p@verts;pts.append(point);ax.scatter(*point,color=C[0],s=35);ax.annotate(f't={t}',point,xytext=label_positions[t],fontsize=10,arrowprops={'arrowstyle':'-','color':'#666','lw':.7})
pts=np.array(pts);ax.plot(pts[:,0],pts[:,1],color=C[0],alpha=.5);p=np.array([.5,.25,.25]);point=p@verts;ax.scatter(*point,color=C[2],marker='s',s=50);ax.annotate('(1/2,1/4,1/4)',point,xytext=(25,-35),textcoords='offset points',arrowprops={'arrowstyle':'->'},color=C[2]);ax.set(aspect='equal',xlim=(-.15,1.15),ylim=(-.12,1.03),title='p0+p1+p2=1；softmax只改变三角形内的位置');ax.axis('off');save('03_simplex.png')

fig,ax=plt.subplots(1,2,figsize=(9.5,3.7));ax[0].axis('off');ax[0].text(.5,.88,'z=(1000,1001,−1000)，y=2',ha='center',fontsize=13);ax[0].text(.5,.57,'减最大值 → (−1,0,−2001)\n概率 → (0.26894,0.73106,0*)',ha='center',linespacing=2,bbox={'boxstyle':'round,pad=.6','fc':'#e9f1f5','ec':'none'});ax[0].text(.5,.19,'直接用logp：损失≈2001.31326\n* 概率下溢；不能再对这个0取log',ha='center',color=C[2],fontsize=11)
ts=np.linspace(0,8,101);direct=[];wrong=[]
for t in ts:
 loss,g,p=multiclass_loss_gradient([[t,0,0]],[0]);direct.append(loss);wrong.append(multiclass_loss_gradient(p,[0])[0])
ax[1].plot(ts,direct,color=C[0],label='CE(logits,y)');ax[1].plot(ts,wrong,color=C[2],label='错误 CE(softmax(logits),y)');ax[1].axhline(math.log(math.e+2)-1,color=C[2],ls=':');ax[1].set(xlabel='z=(t,0,0)，正确类0',ylabel='损失');ax[1].legend(fontsize=8);save('04_logits_loss.png')

p=np.array([.5,.25,.25]);J=np.diag(p)-np.outer(p,p);fig,ax=plt.subplots(1,2,figsize=(8.7,3.3));im=ax[0].imshow(J,cmap='RdBu_r',vmin=-.25,vmax=.25)
for i in range(3):
 for j in range(3):ax[0].text(j,i,f'{J[i,j]:.4f}',ha='center',va='center',color='black',fontsize=12)
ax[0].set(xticks=range(3),yticks=range(3),xlabel='被改变的logit列r',ylabel='概率行k',title='J = diag(p) - p p^T');ax[1].bar(range(3),[-.5,.25,.25],color=[C[2],C[0],C[0]]);ax[1].axhline(0,color='#333');ax[1].set(xticks=range(3),xlabel='类别r',ylabel='单例logit梯度',title='y=0时 p−one_hot(y)',ylim=(-.62,.4));save('05_jacobian.png')

fig,ax=plt.subplots(figsize=(10,3.6));ax.axis('off')
boxes=[(.11,.72,'设计矩阵 Xb\n(N,d+1)'),(.4,.72,'分数 Z=XbW\n(N,K)'),(.76,.72,'沿类别轴softmax\nP 与 Q：(N,K)'),(.76,.22,'Gz=(P−Q)/N\n(N,K)'),(.4,.22,'Gw=Xb^TGz\n(d+1,K)'),(.11,.22,'同步更新 W−ηGw\n(d+1,K)')]
for x,y,t in boxes:ax.text(x,y,t,ha='center',va='center',fontsize=12,bbox={'boxstyle':'round,pad=.7','fc':'#e9f1f5','ec':'none'},linespacing=1.6)
for p0,p1 in [((.22,.72),(.29,.72)),((.51,.72),(.6,.72)),((.76,.57),(.76,.4)),((.64,.22),(.52,.22)),((.29,.22),(.23,.22))]:ax.annotate('',xy=p1,xytext=p0,arrowprops={'arrowstyle':'->','lw':2,'color':C[0]})
ax.text(.4,.48,'W：(d+1,K)\n最后一行是偏置',ha='center',fontsize=10,color=C[2]);save('06_batch_shapes.png')

hist=readcsv('training.csv');w=np.array(result['weights']);fig,ax=plt.subplots(1,2,figsize=(9.5,3.8));steps=[int(r['step']) for r in hist]
ax[0].plot(steps,[float(r['objective']) for r in hist],label='总目标（含正则）',color=C[0]);ax[0].plot(steps,[float(r['mean_cross_entropy']) for r in hist],label='纯均值CE',color=C[1]);ax[0].set(xlabel='更新步',ylabel='损失/目标',title='预定400步、η=0.25、λ=0.03');ax[0].legend(fontsize=9)
g1,g2=np.meshgrid(np.linspace(-3,3,160),np.linspace(-.6,3.6,140));grid=np.column_stack([g1.ravel(),g2.ravel(),np.ones(g1.size)]);pred=np.argmax(grid@w,axis=1).reshape(g1.shape);ax[1].contourf(g1,g2,pred,levels=[-.5,.5,1.5,2.5],cmap=ListedColormap(['#d5eaf0','#f7e5c6','#efdae5']),alpha=.8)
rows=read_data(HERE/'data/classification.csv')
for k,col in enumerate(C[:3]):
 for split,mark in [('train','o'),('test','x')]:
  rs=[r for r in rows if r[1]==split and r[3]==k];ax[1].scatter([r[2][0] for r in rs],[r[2][1] for r in rs],color=col,marker=mark,s=45,label=f'类{k} {split}')
ax[1].set(xlabel='x1',ylabel='x2',title='圆点训练，叉点固定测试');ax[1].legend(fontsize=7,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.23));save('07_training_boundary.png')

br=readcsv('binary_path.csv');fig,ax=plt.subplots(1,2,figsize=(9,3.5));steps=[int(r['step']) for r in br];ws=[float(r['weight']) for r in br];loss=[float(r['mean_bce']) for r in br]
ax[0].plot(steps,ws,color=C[0]);ax[0].set(xlabel='更新步',ylabel='单一权重w',title='无正则且完全可分')
ax[1].semilogy(steps,loss,color=C[2],label='平均BCE');ax[1].plot(steps,[1 if w>0 else .5 for w in ws],color=C[1],label='训练准确率');ax[1].set(xlabel='更新步',ylabel='数值（对数轴）',title='准确率相同，损失仍变化');ax[1].legend();save('08_separation.png')

x=np.array([[-1.,.5,1.],[.2,1.2,1.],[1.,-.8,1.]]);y=np.array([0,2,1]);w=np.array([[.3,-.4,.1],[.2,.1,-.3],[-.1,.2,.1]]);loss,g,p=objective(x,y,w,.03);hs=np.logspace(-1,-10,10);errors=[];saved=None
for h in hs:
 num=np.empty_like(w)
 for idx in np.ndindex(w.shape):
  a=w.copy();b=w.copy();a[idx]+=h;b[idx]-=h;num[idx]=(objective(x,y,a,.03)[0]-objective(x,y,b,.03)[0])/(2*h)
 errors.append(float(np.max(np.abs(num-g))))
 if abs(math.log10(h)+5)<.1:saved=num
fig,ax=plt.subplots(1,2,figsize=(9.5,3.5));ax[0].loglog(hs,errors,'o-',color=C[0]);ax[0].set(xlabel='差分步长h',ylabel='九参数最大绝对误差',title='过大截断、过小抵消')
ax[1].bar(np.arange(9)-.18,g.ravel(),width=.36,color=C[0],label='解析梯度');ax[1].bar(np.arange(9)+.18,saved.ravel(),width=.36,color=C[1],label='h=1e-5中心差分');ax[1].axhline(0,color='#555');ax[1].set(xlabel='按行展平的参数坐标',ylabel='梯度',xticks=range(9));ax[1].legend(fontsize=8);save('09_gradient_check.png')
