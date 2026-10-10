from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/mpl'))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import forward,read_real,fit
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10,'figure.dpi':160})
F=ROOT/'figures';F.mkdir(exist_ok=True)
def save(name):plt.savefig(F/name,bbox_inches='tight');plt.close()
fig,ax=plt.subplots(figsize=(9,2.8));ax.axis('off')
for i,s in enumerate(['物理参数 A,b\n前向曲线 F','仪器增益 g、零点 d\n观测噪声 ε','记录 (x,y)\n条件反演与验证']):
 x=.13+i*.36;ax.text(x,.6,s,ha='center',va='center',bbox=dict(boxstyle='round,pad=.7',facecolor=['#e4edf7','#faecdc','#e9f3e5'][i]),transform=ax.transAxes)
 if i<2:ax.annotate('',xy=(x+.24,.6),xytext=(x+.14,.6),xycoords='axes fraction',arrowprops={'arrowstyle':'->','lw':2})
ax.text(.5,.12,'未标定的增益与幅度只能看到乘积 gA',ha='center',transform=ax.transAxes);save('01_observation.png')
fig,axs=plt.subplots(1,2,figsize=(9,3.3))
for ax,limit in zip(axs,[.05,5]):
 x=np.linspace(0,limit,150)
 for a,b in [(2.4,.55),(1.2,1.1),(4.8,.275)]:ax.plot(x,forward(x,a,b),label=f'A={a}, b={b}')
 ax.set(xlabel='合成无量纲压力 x',ylabel='合成无量纲响应');ax.legend(fontsize=8)
save('02_identifiability.png')
fig,axs=plt.subplots(1,2,figsize=(9,3.4));labels=['narrow','medium','wide']
for k,ax in enumerate(axs):
 arr=[np.loadtxt(ROOT/'outputs'/f'fits_{label}.csv',delimiter=',',skiprows=1)[:,k] for label in labels]
 ax.boxplot(arr,tick_labels=['窄范围','中范围','宽范围'],showfliers=True);ax.axhline([2.4,.55][k],color='#b14945',ls='--',label='已知真值');ax.set_yscale('log');ax.set_ylabel(['幅度 A（对数轴）','速率 b（对数轴）'][k]);ax.legend()
save('03_recovery.png')
data=np.loadtxt(ROOT/'outputs/real_predictions.csv',delimiter=',',skiprows=1);p=data[:,0];v=data[:,1]
fig,axs=plt.subplots(1,2,figsize=(9,3.3));axs[0].scatter(p,v,label='真实观测',s=22);axs[0].plot(p,data[:,2],label='全数据拟合');axs[0].plot(p,data[:,3],ls='--',label='低10点拟合');axs[0].axvline((p[9]+p[10])/2,color='gray',ls=':');axs[0].legend(fontsize=8);axs[0].set_ylabel('原始体积单位')
axs[1].axhline(0,color='gray',lw=1);axs[1].scatter(p,v-data[:,2]);axs[1].set_ylabel('观测减拟合（原始体积单位）')
for ax in axs:ax.set_xlabel('原始压力单位')
save('04_real_data.png')
bs=np.loadtxt(ROOT/'outputs/real_bootstrap.csv',delimiter=',',skiprows=1)
fig,ax=plt.subplots(figsize=(7,3.5));ax.scatter(bs[:,0],bs[:,1],s=9,alpha=.5);ax.set(xlabel='幅度 A（原始体积单位）',ylabel='速率 b（原始压力单位的倒数）',title='300次条件参数自助法，不是300条新观测');ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));save('05_bootstrap.png')
