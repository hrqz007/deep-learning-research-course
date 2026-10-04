"""Original explanatory figures; refuse altered numerical reference files."""
from pathlib import Path
import argparse,csv,hashlib,json,os,tempfile
import numpy as np
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'course-039-mpl'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(tempfile.gettempdir())/'course-039-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import experiment as e
ROOT=Path(__file__).resolve().parent
HASHES={'predictions.csv': 'aa976dd17d7b8f884c469902e655cfd011b4159009d5b81b51b0cae63917844d', 'metrics.csv': '13d1136de69dcd85047664aab312d0421e77e39bf760ae70244f8b0cb45b7d58', 'frozen-plan.json': '761572bcba6f60ce6dc68f1ac20369abc7914b22b8ecde8667d624e9130260ba', 'reliability.csv': '96a8772c925931c06454faecb8ea9b7ae8ea711bd9c2fa259d7607b5d3e6f8c3', 'training-history.csv': 'c993f00d4ab74d127f55462d6cbfd8c0f431ead9693ab60208dd12e0553f2c4a', 'calculations.json': '37b43f04a5ac1b8ee4745555869ec9504a1a05db6e88e81e372fe5ba2fe82031', 'validation-cost-curves.csv': 'f083125cbca276a12c2c59c8226b646ea79d77a11f84b9593dd4f7579247efc8'}
CJK=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if CJK.exists():
    font_manager.fontManager.addfont(str(CJK)); FONT=font_manager.FontProperties(fname=str(CJK)).get_name()
else: FONT='sans-serif'
plt.rcParams.update({'font.family':[FONT,'DejaVu Sans'],'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':13,'figure.dpi':150,'savefig.dpi':160,'axes.unicode_minus':False})
COLORS=['#2563aa','#e77924','#189383'];NAMES={'uniform':'原分布等权','weighted':'类别加权','resampled':'平衡重采样'}

def draw(results,output):
    results=Path(results);output=Path(output)
    for name,h in HASHES.items():
        if hashlib.sha256((results/name).read_bytes()).hexdigest()!=h:raise ValueError(f'{name}: changed result; refusing to rebuild reference figures')
    output.mkdir(parents=True,exist_ok=True)
    read=lambda name:list(csv.DictReader((results/name).read_text().splitlines()))
    ms=read('metrics.csv');bs=read('reliability.csv');hist=read('training-history.csv');curves=read('validation-cost-curves.csv');plan=json.loads((results/'frozen-plan.json').read_text());hand=json.loads((results/'calculations.json').read_text())['first'];made=[]
    def save(fig,name):
        fig.savefig(output/name,bbox_inches='tight',facecolor='white',metadata={'Software':'039 original course figures'});plt.close(fig);made.append(name)
    def select(method,view,group='all',seed=None):return [r for r in ms if r['method']==method and r['view']==view and r['group']==group and (seed is None or int(r['seed'])==seed)]
    fig,ax=plt.subplots(figsize=(10.6,4.2));ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
    nodes=[(.2,2.3,2.5,1.1,'P：原始/部署分布\nη(x)=P(Y=1|x)',COLORS[0]),(3.7,2.3,2.5,1.1,'Q：训练采样分布\n重采样改变出现频率',COLORS[1]),(7.2,2.3,2.5,1.1,'R：优化风险\n损失×权重÷明确分母',COLORS[2]),(3.7,.2,2.5,1.1,'部署：概率 → 行动\n验证校准 + 代价阈值','#705bb5')]
    for x,y,w,h,t,c in nodes:ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.1',fc=c,alpha=.12,ec=c));ax.text(x+w/2,y+h/2,t,ha='center',va='center',color=c)
    for start,end,label in [((2.8,2.85),(3.5,2.85),'选样'),((6.3,2.85),(7,2.85),'评分'),((8.4,2.2),(6.3,.9),'输出须解释'),((1.4,2.2),(3.5,.9),'独立验证来自P')]:
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#536479','lw':1.8});ax.text((start[0]+end[0])/2,(start[1]+end[1])/2+.15,label,fontsize=10,ha='center',bbox={'facecolor':'white','edgecolor':'none','alpha':.92,'pad':1})
    save(fig,'01_three_objects.png')
    p=np.linspace(.001,.999,500);q=4*p/(4*p+1-p)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4));axes[0].plot(p,p,'--',color='gray',label='未加权 q=η');axes[0].plot(p,q,color=COLORS[1],label='a₁/a₀=4');axes[0].scatter([.2],[.5],color=COLORS[1]);axes[0].annotate('真实概率 .2 → 最优加权输出 .5',(.2,.5),(.06,.91),arrowprops={'arrowstyle':'->'},fontsize=10);axes[0].set(xlabel='原分布条件概率 η',ylabel='理想加权风险最优输出 q*',xlim=(0,1),ylim=(0,1));axes[0].legend()
    for weight in [1,4]:axes[1].plot(p,-weight*.2*np.log(p)-.8*np.log(1-p),label=f'a₁={weight}, a₀=1')
    axes[1].set(xlabel='候选输出 q',ylabel='固定 x 的条件加权风险',ylim=(0,3.5));axes[1].legend();save(fig,'02_weighted_posterior.png')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4));loss=np.array([.1,.6,1.3]);a=np.array([1.,2.,3.]);axes[0].bar(np.arange(3)-.18,np.ones(3)/3,width=.36,label='均匀抽样',color=COLORS[0]);axes[0].bar(np.arange(3)+.18,a/6,width=.36,label='qᵢ=aᵢ/Σa',color=COLORS[1]);axes[0].set(xticks=range(3),xticklabels=['样本1','样本2','样本3'],ylabel='抽样概率');axes[0].legend()
    values=[np.dot(a,loss)/3,np.dot(a,loss)/6,np.dot(a/6,loss)];axes[1].bar(['加权/样本数','加权/权重和','重采样期望'],values,color=[COLORS[0],COLORS[1],COLORS[2]]);axes[1].set(ylabel='同一批损失的目标值',ylim=(0,2));
    for i,v in enumerate(values):axes[1].text(i,v+.03,f'{v:.3f}',ha='center')
    save(fig,'03_denominators_sampling.png')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5));axes[0].axis('off');labels=['XW+b → Z','ReLU → H','Hu+c → s','稳定 BCE(s,y)','乘 aᵢ/4 后求和']
    for i,t in enumerate(labels):axes[0].text(.5,.92-i*.2,t,ha='center',va='center',bbox={'boxstyle':'round,pad=.4','fc':['#eaf1fc','#e7f5f2','#eaf1fc','#fff1e8','#ece8fb'][i],'ec':'none'},transform=axes[0].transAxes)
    for i in range(4):axes[0].annotate('',xy=(.5,.79-i*.2),xytext=(.5,.87-i*.2),xycoords='axes fraction',arrowprops={'arrowstyle':'->'})
    paths=np.array(hand['paths']);idx=np.arange(9);axes[1].barh(idx-.17,paths[0],height=.34,label='样本1路径',color=COLORS[0]);axes[1].barh(idx+.17,paths[1],height=.34,label='样本2路径',color=COLORS[1]);axes[1].set(yticks=idx,yticklabels=['W11','W12','W21','W22','b1','b2','u1','u2','c'],xlabel='已含权重和分母的梯度贡献');axes[1].axvline(0,color='gray',lw=.6);axes[1].legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,fontsize=10);save(fig,'04_complete_gradient_paths.png')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4));axes[0].plot(p,1-p,label='告警：1−p',color=COLORS[0]);axes[0].plot(p,4*p,label='不告警：4p',color=COLORS[1]);axes[0].axvline(.2,ls='--',color='gray');axes[0].scatter([.2],[.8],color='black');axes[0].set(xlabel='可信的部署正例概率 p',ylabel='条件期望代价',ylim=(0,2));axes[0].legend()
    axes[1].plot(p,(1-p)/p,color=COLORS[2]);axes[1].axhline(4,ls='--',color='gray');axes[1].axvline(.2,ls='--',color='gray');axes[1].set(xlabel='对应最优阈值 τ',ylabel='FN / FP 的代价比',ylim=(0,10));save(fig,'05_cost_threshold.png')
    s=np.linspace(-4,4,300);offset=np.log(4)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4));axes[0].plot(s,s,label='原始 logit',color=COLORS[0]);axes[0].plot(s,s-offset,label='先验修正：s−ln4',color=COLORS[1]);axes[0].plot(s,s/2,label='温度：s/2',color=COLORS[2]);axes[0].set(xlabel='原始 logit s',ylabel='变换后的 logit');axes[0].legend()
    for t in [.5,1,2]:axes[1].plot(s,e.sigmoid(s/t),label=f'T={t}')
    axes[1].scatter([0],[.5],color='black');axes[1].axhline(.2,ls='--',color=COLORS[1]);axes[1].set(xlabel='原始 logit s',ylabel='σ(s/T)');axes[1].legend();save(fig,'06_shift_vs_temperature.png')
    val=e.load_split(ROOT/'data','validation');mask=val['role']=='calibration';r=next(r for r in plan['runs'] if r['method']=='weighted' and r['seed']==11);z=e.predict(r['params'],val['x'])[mask]-r['prior_logit_subtract'];yy=val['y'][mask];betas=np.linspace(.4,1.5,200)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4));axes[0].plot(betas,[e.bce(b*z,yy).mean() for b in betas],color=COLORS[0]);axes[0].axvline(r['temperature']['beta'],ls='--',color=COLORS[1]);axes[0].set(xlabel='逆温度 β=1/T',ylabel='200条校准验证记录的 NLL');axes[1].plot(betas,[np.mean((e.sigmoid(b*z)-yy)*z) for b in betas],color=COLORS[2]);axes[1].axhline(0,color='gray',lw=.7);axes[1].set(xlabel='逆温度 β',ylabel='d NLL / dβ');save(fig,'07_temperature_fit.png')
    fig,axes=plt.subplots(1,3,figsize=(11,3.7));views=['raw','prior','prior_temperature'];labels=['原始','先验修正','再温度缩放']
    for ax,metric in zip(axes,['nll','brier','accuracy']):
        for j,method in enumerate(e.METHODS):
            for i,view in enumerate(views):
                vals=[float(r[metric]) for r in select(method,view)];ax.scatter(np.full(3,i+(j-1)*.13),vals,c=COLORS[j],s=20,alpha=.7,label=NAMES[method] if i==0 else None)
                ax.plot(i+(j-1)*.13,np.mean(vals),'_',color=COLORS[j],markersize=12)
        ax.set(xticks=range(3),xticklabels=labels,ylabel=metric)
    axes[0].legend(fontsize=9);save(fig,'08_actual_probability_metrics.png')
    fig,axes=plt.subplots(1,3,figsize=(11,3.7))
    for ax,method in zip(axes,e.METHODS):
        ax.plot([0,1],[0,1],'--',color='gray')
        for color,view,label in [(COLORS[1],'raw','原始'),(COLORS[2],'prior_temperature','修正+温度')]:
            rr=[r for r in bs if r['method']==method and r['view']==view and r['group']=='all' and r['seed']=='11' and int(r['n'])>0]
            ax.plot([float(r['mean_p']) for r in rr],[float(r['positive_rate']) for r in rr],color=color,alpha=.7)
            ax.scatter([float(r['mean_p']) for r in rr],[float(r['positive_rate']) for r in rr],s=[10+int(r['n'])*.3 for r in rr],color=color,label=label)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='箱内平均正例概率',ylabel='箱内实际正例比例',title=NAMES[method]);ax.legend(fontsize=9)
    save(fig,'09_reliability_curves.png')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4))
    for j,method in enumerate(e.METHODS):
        rr=[r for r in curves if r['method']==method and r['seed']=='11'];axes[0].plot([float(r['threshold']) for r in rr],[float(r['cost']) for r in rr],'-o',markersize=3,color=COLORS[j],label=NAMES[method])
        for i,view in enumerate(['raw','prior_temperature','cost_validation']):
            vals=[float(r['cost']) for r in select(method,view)];axes[1].scatter(np.full(3,i+(j-1)*.13),vals,color=COLORS[j]);axes[1].plot(i+(j-1)*.13,np.mean(vals),'_',color=COLORS[j],markersize=12)
    axes[0].axvline(.2,color='gray',ls='--');axes[0].set(xlabel='候选概率阈值',ylabel='决策验证集平均代价',title='只用200条决策验证记录选择');axes[0].legend(fontsize=9);axes[1].set(xticks=range(3),xticklabels=['原始 .5','校准 .5','校准+选阈值'],ylabel='测试平均代价',title='测试只评价已经冻结的规则');save(fig,'10_decision_cost_curves.png')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4));groups=['all','0','1']
    for j,method in enumerate(e.METHODS):
        for i,g in enumerate(groups):
            rr=select(method,'prior_temperature',g);axes[0].scatter(np.full(3,i+(j-1)*.13),[float(r['ece']) for r in rr],color=COLORS[j],label=NAMES[method] if i==0 else None)
        rr=select(method,'prior_temperature','all');axes[1].plot([5,10,20],[np.mean([float(r[c]) for r in rr]) for c in ['ece5','ece','ece20']],'-o',color=COLORS[j],label=NAMES[method])
    axes[0].set(xticks=range(3),xticklabels=['全体 n=1000','组0 n=731','组1 n=269'],ylabel='10等宽箱 ECE',title='总体校准不保证每组校准');axes[0].legend(fontsize=9);axes[1].set(xticks=[5,10,20],xlabel='等宽分箱数',ylabel='三种子平均 ECE',title='分箱规则改变数值');axes[1].legend(fontsize=9);save(fig,'11_groups_and_bins.png')
    grid=np.stack(np.meshgrid(np.linspace(-2.5,2.5,160),np.linspace(-2.5,2.5,160)),axis=-1);flat=grid.reshape(-1,2)
    for num,mode in [(12,'raw'),(13,'calibrated')]:
        fig,axes=plt.subplots(1,3,figsize=(11,3.6))
        for ax,method in zip(axes,e.METHODS):
            r=next(r for r in plan['runs'] if r['method']==method and r['seed']==11);zz=e.predict(r['params'],flat)
            if mode=='calibrated':zz=r['temperature']['beta']*(zz-r['prior_logit_subtract'])
            pp=e.sigmoid(zz).reshape(grid.shape[:2]);im=ax.contourf(grid[:,:,0],grid[:,:,1],pp,levels=np.linspace(0,1,11),cmap='RdYlBu_r',vmin=0,vmax=1);ax.contour(grid[:,:,0],grid[:,:,1],pp,levels=[.2,.5],colors=['#141a29','#ffffff'],linewidths=[1.8,1.8]);ax.set(xlabel='x₁',ylabel='x₂',title=NAMES[method]);ax.text(.03,.03,'黑线 .2；白线 .5',transform=ax.transAxes,fontsize=9,bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.85,label='输出正例分数',pad=.025);save(fig,f'{num:02}_decision_surfaces.png')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4));axes[0].scatter([.1,.1,.9,.9],[0,1,0,1],s=90,color=COLORS[0]);axes[0].plot([0,1],[0,1],'--',color='gray');axes[0].set(xlabel='四条预测概率',ylabel='观测标签',xlim=(0,1),ylim=(-.1,1.1));axes[1].bar(['1个箱','2个箱'],[0,.4],color=[COLORS[2],COLORS[1]]);axes[1].set(ylabel='这个有限样本的 ECE',ylim=(0,.5));axes[1].text(0,.015,'0 ≠ 证明完全校准',ha='center',fontsize=10);axes[1].text(1,.415,'0.4',ha='center');save(fig,'14_bin_cancellation.png')
    return {name:{'sha256':hashlib.sha256((output/name).read_bytes()).hexdigest(),'bytes':(output/name).stat().st_size} for name in made}

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=ROOT/'outputs');p.add_argument('--output',type=Path,default=ROOT/'figures');a=p.parse_args();print(json.dumps(draw(a.results,a.output),indent=2))
if __name__=='__main__':main()
