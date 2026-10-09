"""由真实outputs和明示数学示例重建原创彩色图，不下载素材。"""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/course-dl-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent
font=FontProperties(fname=os.environ.get('COURSE_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
C=['#2563eb','#e76f51','#10a380','#9b5de5']
R=json.loads((ROOT/'outputs/results.json').read_text())
F=ROOT/'figures';F.mkdir(exist_ok=True)
def save(name):
    plt.tight_layout();plt.savefig(F/name,dpi=170,bbox_inches='tight');plt.close()
def flow(labels,name):
    fig,ax=plt.subplots(figsize=(10,2.3));ax.axis('off')
    for i,s in enumerate(labels):
        xx=.12+i*.255
        ax.text(xx,.52,s,ha='center',va='center',fontsize=11,bbox=dict(boxstyle='round,pad=.65',fc=['#dbeafe','#d1fae5','#ffedd5','#ede9fe'][i],ec=C[i]))
        if i<3:ax.annotate('',xy=(xx+.17,.52),xytext=(xx+.08,.52),arrowprops=dict(arrowstyle='->',color='#475569',lw=2))
    save(name)

A=np.load(ROOT/'outputs/trajectories.npz')
flow(['初值theta0\nf0与Jacobian J0','两条训练路径\n真实网络 / 线性化','同数据同初值步长\n扫描宽度和目标幅度','对比预测与核漂移\n记录近似失效'],'01_comparison.png')
fig,ax=plt.subplots(1,2,figsize=(10,3.5))
for k,key in enumerate(['w8_a4_s11','w128_a0.2_s11']):
    amp=4 if k==0 else .2;ax[k].plot(A['grid'],amp*A['grid_y'],color='#334155',label='目标');ax[k].plot(A['grid'],A[key+'_nonlinear_grid'],color=C[0],label='真实网络');ax[k].plot(A['grid'],A[key+'_linear_grid'],'--',color=C[1],label='线性化');ax[k].set(title=key,xlabel='x',ylabel='预测');ax[k].legend(fontsize=8)
save('02_predictions.png')
fig,ax=plt.subplots(1,2,figsize=(10,3.5))
for key,c in [('w8_a4_s11',C[1]),('w128_a4_s11',C[0]),('w128_a0.2_s11',C[2])]:
    h=A[key+'_history'];ax[0].semilogy(h[:,0],np.maximum(h[:,1],1e-12),color=c,label=key+'真实');ax[0].semilogy(h[:,0],np.maximum(h[:,2],1e-12),'--',color=c);ax[1].plot(h[:,0],h[:,3],color=c,label=key)
ax[0].set(xlabel='更新次数',ylabel='平均平方损失/2；虚线为线性化');ax[0].legend(fontsize=7);ax[1].set(xlabel='更新次数',ylabel='预测RMSE / 目标幅度');ax[1].legend(fontsize=7);save('03_trajectories.png')
fig,ax=plt.subplots(1,2,figsize=(9,3.6))
for j,field,title in [(0,'relative_prediction_rmse','平均相对预测差'),(1,'kernel_drift','平均核相对漂移')]:
    z=np.array([[np.mean([r[field] for r in R['rows'] if r['width']==w and r['amplitude']==a]) for w in [8,32,128]] for a in [.2,1.,4.]])
    im=ax[j].imshow(z,cmap='YlOrRd',aspect='auto');ax[j].set(xticks=range(3),xticklabels=[8,32,128],yticks=range(3),yticklabels=[.2,1,4],xlabel='宽度m',ylabel='目标幅度A',title=title)
    for i in range(3):
        for k in range(3):ax[j].text(k,i,f'{z[i,k]:.3f}',ha='center',va='center',color='white' if z[i,k]>z.max()*.6 else 'black')
    fig.colorbar(im,ax=ax[j],shrink=.75)
save('04_scan.png')
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
for amp,c in [(.2,C[2]),(1.,C[0]),(4.,C[1])]:
    rows=[r for r in R['rows'] if r['amplitude']==amp];ax[0].scatter([r['kernel_drift'] for r in rows],[r['relative_prediction_rmse'] for r in rows],c=c,label='A='+str(amp));ax[1].scatter([r['feature_drift'] for r in rows],[r['relative_parameter_movement'] for r in rows],c=c,label='A='+str(amp))
ax[0].set(xlabel='核相对漂移',ylabel='相对预测差');ax[1].set(xlabel='隐藏表示相对漂移',ylabel='相对参数移动');ax[0].legend();ax[1].legend();save('05_drift.png')
