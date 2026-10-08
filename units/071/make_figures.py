"""Six original figures. Empirical panels read saved results, never invented data."""
from pathlib import Path
import json,os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'tmp/mpl'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(__file__).resolve().parent/'tmp/cache'))
cache=Path(__file__).resolve().parent/'tmp/cache'
cache.mkdir(parents=True,exist_ok=True)
os.environ['XDG_CACHE_HOME']=str(cache)
fc=cache/'fontconfig.xml'
fc.write_text('<fontconfig><include ignore_missing="yes">/etc/fonts/fonts.conf</include><cachedir>'+str(cache/'fontconfig')+'</cachedir></fontconfig>')
(cache/'fontconfig').mkdir(exist_ok=True)
os.environ['FONTCONFIG_FILE']=str(fc)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parent
font=os.environ.get('COURSE_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if Path(font).exists():
    font_manager.fontManager.addfont(font);plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'figure.facecolor':'white'})
COLORS=['#2369a2','#df7e26','#459b85','#ae4a67','#8061a8']
NAMES={'id':'同分布 ID','core_noise':'核心加噪','shortcut_fade':'捷径衰减','shortcut_flip':'捷径反转','subgroup_flip':'少数群反转'}

def save(fig,name):
    (ROOT/'figures').mkdir(exist_ok=True);fig.savefig(ROOT/'figures'/name,dpi=165,bbox_inches='tight');plt.close(fig)

def main():
    d=np.load(ROOT/'outputs/data.npz');p=np.load(ROOT/'outputs/predictions.npz');r=json.loads((ROOT/'outputs/results.json').read_text());run=r['runs'][0]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in zip(axes,['train_x','shortcut_flip_x'],['训练域：两个特征都与标签同向','反转域：核心规律保留，捷径反向']):
        y=d['train_y'] if key=='train_x' else d['test_y'];x=d[key]
        for cls,c in zip([0,1],COLORS):ax.scatter(x[:250][y[:250]==cls,0],x[:250][y[:250]==cls,1],c=c,s=13,alpha=.65,label=f'真实类 {cls}')
        ax.set(xlabel='核心特征 x0',ylabel='捷径特征 x1',title=title,xlim=(-4.5,4.5),ylim=(-4,4));ax.legend(fontsize=9)
    fig.tight_layout();save(fig,'01_domains.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    cases={'成员均为 [0.5, 0.5]':np.array([[.5,.5],[.5,.5]]),'成员相反且各自自信':np.array([[.99,.01],[.01,.99]])}
    from experiment import entropy
    for ax,(title,stack) in zip(axes,cases.items()):
        h=float(entropy(stack.mean(0)));a=float(entropy(stack).mean());mi=h-a
        ax.bar(['总预测熵','平均成员熵','分歧差'],[h,a,mi],color=COLORS[:3]);ax.set_ylim(0,.8);ax.set_ylabel('熵 / nat');ax.set_title(title)
        for i,v in enumerate([h,a,mi]):ax.text(i,v+.025,f'{v:.3f}',ha='center')
    fig.suptitle('数学示例：相同的总熵可能来自不同成员行为',fontsize=14);fig.tight_layout();save(fig,'02_uncertainty.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,name in zip(axes,['id','shortcut_flip']):
        for label,key,c in [('原始','raw_reliability',COLORS[0]),('温度后','calibrated_reliability',COLORS[1])]:
            rows=run['domains'][name][key]['bins'];ax.scatter([v['confidence'] for v in rows],[v['accuracy'] for v in rows],s=[max(30,v['n']*.25) for v in rows],color=c,label=label,alpha=.8)
        ax.plot([0,1],[0,1],'--',color='#777',lw=1);ax.set(xlim=(0,1.04),ylim=(-.04,1.04),xlabel='箱内平均置信度',ylabel='箱内准确率',title=NAMES[name]);ax.legend()
    fig.suptitle('实际结果：圆面积与箱内样本量相关，空箱不画',fontsize=13);fig.tight_layout();save(fig,'03_reliability.png')
    fig,ax=plt.subplots(figsize=(8.8,4.2))
    for name,c in zip(NAMES,COLORS):
        values=run['domains'][name]['risk_coverage'];ax.plot(values['coverage'],values['risk'],label=NAMES[name],color=c,lw=2)
        s=run['domains'][name]['selective']
        if s['risk'] is not None:ax.scatter(s['coverage'],s['risk'],color=c,s=50,edgecolors='black',zorder=4)
    ax.set(xlim=(0,1.01),ylim=(-.035,1.035),xlabel='覆盖率：被接受样本 / 全部样本',ylabel='选择性风险：接受后的错误率',title='seed 17 实测，圆点为同一个校准阈值');ax.legend(ncol=2,fontsize=10);fig.tight_layout();save(fig,'04_risk_coverage.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    groups=run['domains']['subgroup_flip']['groups'];xs=np.arange(2)
    for ax,metric,title in zip(axes,['coverage','risk'],['每组覆盖率','每组接受后错误率']):
        values=[groups[str(g)][metric] for g in [0,1]];ax.bar(xs,values,color=COLORS[:2]);ax.set_xticks(xs,[f'g={g}\nn={groups[str(g)]["n"]}' for g in [0,1]]);ax.set_ylim(0,1.1);ax.set_title(title)
        for i,v in enumerate(values):ax.text(i,v+.025,f'{v:.1%}',ha='center')
    fig.suptitle('少数群反转：总体平均可以遮住子群失败',fontsize=14);fig.tight_layout();save(fig,'05_subgroups.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,name in zip(axes,['id','shortcut_flip']):
        prob=p[f'seed17_{name}_p'];wrong=prob.argmax(1)!=d['test_y'];conf=prob.max(1)
        ax.hist([conf[~wrong],conf[wrong]],bins=np.linspace(.5,1,21),stacked=True,color=[COLORS[2],COLORS[3]],label=['正确','错误']);ax.axvline(.95,color='#555',ls='--');ax.set(xlabel='温度后最大概率',ylabel='样本数',title=f'{NAMES[name]}：≥0.95 且错 {int((wrong&(conf>=.95)).sum())} 个');ax.legend()
    fig.tight_layout();save(fig,'06_confident_wrong.png')

if __name__=='__main__':main()
