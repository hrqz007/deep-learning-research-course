from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parent;os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/mpl'))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':['Noto Sans CJK JP','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
from experiment import exact,euler
D=np.load(ROOT/'outputs/data.npz');F=ROOT/'figures';F.mkdir(exist_ok=True)
def save(n):plt.tight_layout();plt.savefig(F/n,dpi=180,bbox_inches='tight');plt.close()
fig,ax=plt.subplots(figsize=(8,4));t=np.linspace(0,2,200);ax.plot(t,exact(1,2,t),label='解析解');tt,yy,_=euler(n=4);ax.plot(tt,yy,'o--',label='Euler 4步');ax.plot(D['t'],D['euler_y'],'o-',ms=3,label='Euler 16步');ax.set(xlabel='时间 / s',ylabel='状态',title='同一方程，离散步长改变数值轨迹');ax.legend();save('01_trajectory.png')
fig,axes=plt.subplots(1,3,figsize=(11,3.5))
for ax,y,dt in zip(axes,D['stability_trajectories'],[.5,1.5,2.5]):ax.plot(np.arange(13),y,'o-');ax.axhline(0,c='#94a3b8',lw=.7);ax.set(xlabel='离散步数',ylabel='状态',title=f'hk={dt}')
save('02_stability.png')
fig,ax=plt.subplots(figsize=(8,4));s=D['scan'];ax.loglog(s[:,1],s[:,2],'o-',label='Euler状态误差');ax.loglog(s[:,1],s[:,4],'s-',label='RK4状态误差');ax.set(xlabel='时间步长 h / s',ylabel='终点绝对误差');ax.legend();save('03_convergence.png')
fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].plot(D['t'],D['sensitivity'],'o-',label='离散敏感度');axes[0].plot(D['t'],-D['t']*D['exact_y'],'--',label='连续解析敏感度');axes[0].set(xlabel='时间 / s',ylabel='∂y/∂k');axes[0].legend();axes[1].loglog(s[:,1],s[:,3],'o-',label='对连续梯度误差');axes[1].loglog(s[:,1],np.maximum(s[:,5],1e-16),'s-',label='对离散差分误差');axes[1].set(xlabel='时间步长 h / s',ylabel='绝对差');axes[1].legend();save('04_gradient.png')
fig,ax=plt.subplots(figsize=(8,4));f=D['fit_scan'];ax.plot(f[:,0],f[:,1],'o-',label='Euler拟合');ax.axhline(1,c='#dc2626',ls='--',label='连续真实k');ax.set(xlabel='积分步数',ylabel='估计衰减率 / s⁻¹',title='训练终值拟合好，不代表参数无离散偏差');ax.legend();save('05_inverse_bias.png')
fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].plot(D['heat_x'],D['heat_exact'],label='解析');axes[0].plot(D['heat_x'],D['heat_u'],'o',ms=3,label='显式差分');axes[0].set(xlabel='位置 / m',ylabel='温度差（统一单位）',title='t=0.25 s，冷端固定为0');axes[0].legend();axes[1].plot(D['heat_x'],D['heat_u']-D['heat_exact']);axes[1].set(xlabel='位置 / m',ylabel='数值减解析',title='μ=0.4，最大误差约1.39e-4');save('06_heat.png')
