"""Original educational diagrams and measured project results."""
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
COLORS=['#62849a','#d67738','#26978b','#a4648f','#6c6fc1']
NAMES={'linear':'线性基线','erm':'MLP 基线','random_corner':'角标随机化','wrong_corner':'错位随机化','mask_corner':'遮挡角标'}

def save(fig,name):
    (ROOT/'figures').mkdir(exist_ok=True);fig.savefig(ROOT/'figures'/name,dpi=165,bbox_inches='tight');plt.close(fig)

def boxes(ax,items):
    ax.axis('off')
    for x,y,text,c in items:ax.text(x,y,text,ha='center',va='center',transform=ax.transAxes,fontsize=11,bbox={'boxstyle':'round,pad=.7','facecolor':c,'edgecolor':'none'})

def main():
    r=json.loads((ROOT/'outputs/results.json').read_text());d=np.load(ROOT/'outputs/data.npz')
    fig,ax=plt.subplots(figsize=(10,4.2));boxes(ax,[(.17,.77,'问题\n打乱角标能否减少捷径依赖？','#dfedf7'),(.5,.77,'主张\n预定反转测试错误率下降','#dff1e9'),(.83,.77,'证据\n同样本 配对种子 消融','#fff0d6'),(.26,.23,'替代解释\n一般噪声？容量？信息删除？','#f6e4ed'),(.73,.23,'适用边界\n已知角标位置 人造图像','#e9e5f6')])
    for x1,y1,x2,y2 in [(.34,.77,.39,.77),(.65,.77,.72,.77),(.5,.64,.3,.39),(.58,.64,.72,.39)]:ax.annotate('',(x2,y2),(x1,y1),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#667587','lw':1.5})
    ax.set_title('概念图：每个研究主张都要能连回证据与边界',pad=8);save(fig,'01_claim_map.png')
    fig,axes=plt.subplots(2,3,figsize=(9,5.2))
    for row,cls in enumerate([0,1]):
        idx=int(np.flatnonzero(d['test_y']==cls)[0])
        for ax,key,title in zip(axes[row],['id_x','flip_x','blank_x'],['同分布角标','反转角标','空白角标']):
            ax.imshow(d[key][idx,0],vmin=-2.2,vmax=2.2,cmap='RdBu_r');ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f'类 {cls} · {title}',fontsize=11)
    fig.suptitle('实际合成样本：标签由中心线条方向定义，角标是附加线索',fontsize=13);fig.tight_layout();save(fig,'02_data.png')
    fig,ax=plt.subplots(figsize=(10,3.8));boxes(ax,[(.17,.67,'训练 800\n只更新模型参数','#dfeef8'),(.50,.67,'验证 240\n只定 80% 覆盖阈值','#e2f2e6'),(.83,.67,'封存测试 600\n四域配对评估','#fff0d8'),(.50,.18,'固定 5 方法 × 3 种子 × 160 步\n不按测试结果挑方法或超参数','#eee9f7')]);ax.set_title('实验数据边界：种子固定，角色分开',pad=10);save(fig,'03_protocol.png')
    fig,ax=plt.subplots(figsize=(10,4.5));xs=np.arange(4);width=.16
    for j,(v,c) in enumerate(zip(NAMES,COLORS)):
        a=r['aggregate'][v]['domains'];means=[a[k]['mean_error'] for k in ['id','flip','blank','core_noise']];sd=[a[k]['sd_error'] for k in ['id','flip','blank','core_noise']]
        ax.bar(xs+(j-2)*width,means,width,yerr=sd,capsize=3,color=c,label=NAMES[v])
    ax.set_xticks(xs,['同分布','角标反转','角标空白','核心加噪']);ax.set(ylabel='错误率',ylim=(0,.87),title='实际 3 种子均值 ± 样本标准差（同一固定数据集）');ax.legend(ncol=3,fontsize=10);fig.tight_layout();save(fig,'04_results.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4.3));delta=r['paired_difference']['by_seed'];axes[0].bar([str(s) for s in r['config']['seeds']],delta,color=COLORS[2]);axes[0].axhline(0,color='#333',lw=1);axes[0].set(ylim=(0,.65),xlabel='配对初始化种子',ylabel='基线错误率 − 干预错误率',title='反转域的配对改善量')
    for i,v in enumerate(delta):axes[0].text(i,v+.02,f'{v:.3f}',ha='center')
    for v,c in zip(['erm','random_corner','wrong_corner'],[COLORS[1],COLORS[2],COLORS[3]]):
        run=next(x for x in r['runs'] if x['seed']==11 and x['variant']==v);cu=run['risk_coverage']['flip'];axes[1].plot(cu['coverage'],cu['risk'],color=c,label=NAMES[v])
    axes[1].set(xlim=(0,1),ylim=(0,1),xlabel='覆盖率',ylabel='接受后的错误率',title='反转域拒答曲线（seed 11）');axes[1].legend(fontsize=9);fig.tight_layout();save(fig,'05_paired_selective.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4.4))
    for v,c in zip(NAMES,COLORS):
        a=r['aggregate'][v];axes[0].scatter(a['train_seconds_mean'],a['domains']['flip']['mean_error'],color=c,s=80,label=NAMES[v]);axes[1].bar(NAMES[v],a['parameters'],color=c)
    axes[0].set(xlabel='每模型优化循环平均秒数',ylabel='反转域平均错误率',title='本机实测预算与结果');axes[0].legend(fontsize=9);axes[1].set(ylabel='可训练参数个数',title='相同步数不代表同等算力');axes[1].tick_params(axis='x',rotation=25,labelsize=9);fig.tight_layout();save(fig,'06_budget.png')

if __name__=='__main__':main()
