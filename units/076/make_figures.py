"""Rebuild six original color figures from local saved evidence."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/course-matplotlib')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager
CJK=os.environ.get('COURSE_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
font_manager.fontManager.addfont(CJK)
FONT=font_manager.FontProperties(fname=CJK).get_name()
plt.rcParams.update({'font.family':FONT,'axes.unicode_minus':False,'font.size':10})
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
BLUE='#1768ac'; ORANGE='#dc772a'; GREEN='#23856d'; RED='#b54555'
def save(fig,name):
    fig.tight_layout()
    fig.savefig(OUT/name,dpi=160,bbox_inches='tight')
    plt.close(fig)
def box(ax,x,y,text,color=BLUE,w=.24,h=.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.018',fc=color,ec='none',alpha=.13))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',color=color,fontsize=11)
def arrow(ax,a,b):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#526075','lw':1.7})

d=np.load(ROOT/'outputs/data.npz'); p=np.load(ROOT/'outputs/predictions.npz'); r=json.loads((ROOT/'outputs/results.json').read_text())
fig,ax=plt.subplots(figsize=(9,3.2)); ax.axis('off'); ax.set(xlim=(0,1),ylim=(0,1))
box(ax,.03,.4,'先验 p(θ)\n观察数据 D');box(ax,.38,.4,'更新参数分布\np(θ | D)',GREEN);box(ax,.73,.4,'每个 θ 预测\n再对预测取平均',ORANGE)
arrow(ax,(.29,.5),(.36,.5));arrow(ax,(.64,.5),(.71,.5));ax.text(.5,.14,'参数不确定性通过函数传播；观测噪声在预测端另外加入',ha='center',color=RED)
save(fig,'01_prediction_flow.png')
fig,ax=plt.subplots(figsize=(9,3.5));g=d['grid_x'];ax.plot(g,d['truth'],color=GREEN,label='已知生成均值（仅评价）');ax.scatter(d['train_x'],d['train_y'],s=13,color=BLUE,label='100个训练观测');ax.axvspan(-2,2,color=BLUE,alpha=.08,label='训练支持区间');ax.set(xlabel='输入 x',ylabel='结果 y');ax.legend(ncol=2);save(fig,'02_data_support.png')
fig,axes=plt.subplots(1,3,figsize=(11,3.5),sharex=True,sharey=True)
for k,(ax,title,c) in enumerate(zip(axes,['单网络','五成员集成','固定特征最后层 Bayes'],[BLUE,ORANGE,GREEN])):
 m=p['means'][k];s=np.sqrt(p['variances'][k]);ax.fill_between(g,m-1.96*s,m+1.96*s,color=c,alpha=.2);ax.plot(g,m,color=c,label='预测均值');ax.plot(g,d['truth'],'--',color='#111827',label='生成均值');ax.axvspan(-2,2,color='#94a3b8',alpha=.12);ax.set(title=title,xlabel='x',ylim=(-4,4))
axes[0].set_ylabel('y 与名义95%预测带');axes[-1].legend(fontsize=8);save(fig,'03_prediction_bands.png')
fig,axes=plt.subplots(1,2,figsize=(9,3.4));
for k,c,label in zip(range(3),[BLUE,ORANGE,GREEN],['单网络','集成','最后层Bayes']):
 axes[0].plot(g,p['variances'][k]-0.18**2,color=c,label=label)
axes[0].set(xlabel='x',ylabel='参数/成员分歧项（方差）');axes[0].legend(fontsize=8)
axes[1].plot(g,p['mc_mean']-p['means'][2],color=GREEN);axes[1].axhline(0,color='#777',lw=.8);axes[1].set(xlabel='x',ylabel='4000次采样均值减解析均值');save(fig,'04_variance_mc.png')
fig,axes=plt.subplots(1,2,figsize=(9,3.4));names=['single','ensemble','last_layer_bayes'];labels=['单网络','集成','最后层Bayes'];loc=np.arange(3)
for offset,region,c,lab in [(-.22,'in_domain',BLUE,'域内'),(0,'out_of_domain',ORANGE,'全部域外'),(.22,'far_out',RED,'远域 |x|≥4')]:
 axes[0].bar(loc+offset,[r[n][region]['coverage_95'] for n in names],.22,color=c,label=lab)
 axes[1].bar(loc+offset,[r[n][region]['mean_width_95'] for n in names],.22,color=c)
axes[0].axhline(.95,color='black',ls='--',lw=1);axes[0].set(ylabel='实测覆盖率',ylim=(0,1.15));axes[0].legend(fontsize=8,ncol=1);axes[1].set(ylabel='平均区间宽度')
for ax in axes:ax.set_xticks(loc,labels)
save(fig,'05_coverage_width.png')
fig,ax=plt.subplots(figsize=(9,3.5));
for i,seed in enumerate([11,23,37,53,71]):ax.plot(g,p['members'][i],lw=1,label=f'成员 {seed}')
ax.plot(g,d['truth'],'--',color='black',lw=2,label='生成均值');ax.axvspan(-2,2,color=BLUE,alpha=.08);ax.set(xlabel='x',ylabel='各网络均值',title='共享 tanh 归纳偏好：成员可以一致地外推错误');ax.legend(ncol=3,fontsize=8);save(fig,'06_shared_failure.png')
