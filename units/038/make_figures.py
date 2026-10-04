"""Rebuild twelve original figures from hash-checked public experiment outputs."""
import argparse,csv,hashlib,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/dl038-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/dl038-cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':FONT.get_name(),'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':160})
COLORS=['#2563eb','#e87524','#0d9488','#9333ea','#dc2626']
EXPECTED={'outputs/capacity-detail.json': 'bc641daa28438ea552cea2d103e8fc322fc1c6b60e9557c4b22e8dcd1e445b1e', 'outputs/capacity.csv': 'c701a6318227f9cc6352f06074475deee6c0d0735427e94a9b379ea426c89ff8', 'outputs/conditional-risk.json': '1ad15ff09fc4e15a380298a4d45a12594d6cada0296b6f42c8351fe3691dc757', 'outputs/duplication.json': 'b28a023672dd983287e04ef271d6eefeb164f3b348a84212f6e9ec04ef33a1e5', 'outputs/finite-class-bound.json': '6b17be79d708d83462049bba64f017f488d006d59f72e7440edb9139ef920221', 'outputs/hand-trace.json': 'f5d65226cc0d8caa795b2c5005cdeb58fc7df393516fe62d4ebea4bd36752637', 'outputs/implicit-bias.json': '8c243fe95f65c35895fd1f69b1f8c9a1da36a8c79ce70fe46f227abe66f60895', 'outputs/spectral-filters.json': 'ac1d2d30f75d43c3893aae376da487a41e7307ba5474c3945e80a16a5cc3acbb', 'outputs/summary.json': '6e18ec0e444a6983e7d784695911302b54f1e0f8e0b9b58947e3f186466b26ba', 'data/gaussian-fixture.json': '5918bdbd00997445ba621d8c0b95e989042535bb6bb5e6c93ed3eb89ddb8a90e', 'experiment.py': 'fa34627aed0d37ac59fdf016508c3b2c560f6e79ca8d6d33454f135910f9c48e', 'test_experiment.py': '289aa2990cb5567c8177b917aab52bc41e817b54dbede8e4d136f6605b0fac28'}
def read_inputs():
    for name,h in EXPECTED.items():
        p=ROOT/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError('Figure input changed: '+name)
    out=ROOT/'outputs';rows=list(csv.DictReader((out/'capacity.csv').open()))
    for r in rows:
        for k in r:r[k]=float(r[k])
    docs={p.stem:json.loads(p.read_text()) for p in out.glob('*.json')}
    return rows,docs

def finish(fig,out,name):
    fig.tight_layout(pad=1.5);fig.savefig(out/name,facecolor='white');plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'figures');args=parser.parse_args();rows,d=read_inputs();out=args.output;out.mkdir(parents=True,exist_ok=True)
    widths=sorted(set(int(r['p']) for r in rows))
    def values(p,sigma,lam,key):return np.array([r[key] for r in rows if r['p']==p and r['sigma']==sigma and r['ridge']==lam])
    def curve(ax,sigma,lam,key,label,color,individual=False):
        if individual:
            for seed in range(3800,3812):
                yy=[next(r[key] for r in rows if r['seed']==seed and r['p']==p and r['sigma']==sigma and r['ridge']==lam) for p in widths];ax.plot(widths,np.maximum(yy,1e-12),color=color,alpha=.17,lw=.75)
        ax.plot(widths,[max(np.median(values(p,sigma,lam,key)),1e-12) for p in widths],'o-',color=color,label=label,lw=2,ms=3)
    # 01 A ladder, with three genuinely different notions.
    fig,ax=plt.subplots(figsize=(10.5,3.8));ax.axis('off')
    labels=[('参数坐标 θ','数量多\n可能含冗余'),('可表达函数 F','架构与约束\n决定候选集合'),('算法选中的函数','数据、初始化、\n优化器、停止规则'),('目标分布上的风险','还需要新的输入\n与评价协议')]
    for i,(a,b) in enumerate(labels):
        x=.02+i*.25;ax.text(x+.10,.6,a,ha='center',va='center',color=COLORS[i],weight='bold',fontsize=14,bbox=dict(boxstyle='round,pad=.6',fc='#eff6ff',ec=COLORS[i]));ax.text(x+.1,.28,b,ha='center',va='center',linespacing=1.8)
        if i<3:ax.annotate('',xy=(x+.24,.6),xytext=(x+.215,.6),arrowprops={'arrowstyle':'->','lw':1.5})
    finish(fig,out,'01-capacity-levels.png')
    fig,axs=plt.subplots(1,2,figsize=(10.5,4))
    for ax,s,t in zip(axs,d['hand-trace']['steps'],['第1步','第2步']):
        a=np.array(s['contributions']);ax.imshow(a,cmap='coolwarm',vmin=-1,vmax=1,aspect='auto');ax.set(xticks=range(3),xticklabels=['w1','w2','w3'],yticks=[0,1],yticklabels=['样本1','样本2'],title=t+'：每个样本对平均损失梯度的贡献')
        for i in range(2):
            for j in range(3):ax.text(j,i,f'{a[i,j]:.4g}',ha='center',va='center',color='black',fontsize=13)
        ax.set_xlabel('按列求和 = '+str(s['gradient']))
    finish(fig,out,'02-gradient-contributions.png')
    fig,axs=plt.subplots(1,2,figsize=(10.5,4));t=np.linspace(-2,2,201);axs[0].plot(t,2+3*t*t,color=COLORS[0],label='参数平方范数');axs[0].plot(t,np.zeros_like(t),color=COLORS[2],label='训练平方误差');axs[0].scatter([0],[2],color=COLORS[0]);axs[0].set(xlabel='零空间系数 t',ylabel='数值',title='所有 w*= (0,1,1) + t(-1,-1,1) 都插值');axs[0].legend()
    for p in d['implicit-bias']['paths']:
        w=np.array([r['w'] for r in p['rows']]);axs[1].plot(range(81),w@np.array([1,1,0]),label='初始 t = '+str(p['null_coefficient']))
    axs[1].set(xlabel='GD 更新次数',ylabel='新点 (1,1,0) 的预测',title='相同训练数据不能识别零空间分量');axs[1].legend();finish(fig,out,'03-null-space.png')
    fig,axs=plt.subplots(1,2,figsize=(10.5,4));s=np.geomspace(.03,3,300)
    for t,c in zip([1,10,100,1000],COLORS):axs[0].plot(s,(1-(1-.1*s*s)**t)/s,label=f'GD t={t}',color=c)
    axs[0].plot(s,1/s,'k--',label='完全最小二乘 1/s');axs[0].set(xscale='log',yscale='log',xlabel='奇异值 s',ylabel='乘到 u^T y 上的系数',title='η/n = 0.1，有限步抑制小奇异值');axs[0].legend(fontsize=9)
    for lam,c in zip([.001,.01,.1,1],COLORS):axs[1].plot(s,s/(s*s+2*lam),label=f'λ={lam}',color=c)
    axs[1].plot(s,1/s,'k--',label='无惩罚');axs[1].set(xscale='log',yscale='log',xlabel='奇异值 s',ylabel='Ridge 系数 s/(s²+2λ)',title='n=2：滤波形状与早停不同');axs[1].legend(fontsize=9);finish(fig,out,'04-spectral-filter.png')
    fig,axs=plt.subplots(1,2,figsize=(10.5,4));x=np.arange(2);a,b=d['duplication']['models'];axs[0].bar(x-.18,a['prediction'],.35,label='6参数');axs[0].bar(x+.18,b['prediction'],.35,label='11参数');axs[0].set(xticks=x,xticklabels=['样本1','样本2'],ylabel='预测',title='复制隐藏节点、平分输出权重：初始函数相同');axs[0].legend()
    axs[1].bar(x-.18,a['next_prediction'],.35,label='6参数更新后');axs[1].bar(x+.18,b['next_prediction'],.35,label='11参数更新后');axs[1].scatter(x,[1,2],marker='x',s=65,color='black',label='标签');axs[1].set(xticks=x,xticklabels=['样本1','样本2'],ylabel='预测',title='同一学习率0.2：一步后函数不同');axs[1].legend(fontsize=9);finish(fig,out,'05-neuron-duplication.png')
    fig,ax=plt.subplots(figsize=(10.5,3.8));ax.axis('off')
    blocks=[('12个配对seed','每个固定 X(24×96)\n和同一噪声 ε'),('18个嵌套宽度','只取前 p 列\np=1 到96'),('3个噪声强度','σ=0、0.2、0.8\ny=Xβ+σε'),('3个固定惩罚','λ=0、0.01、0.1\n共1944个拟合')]
    for i,(a,b) in enumerate(blocks):
        ax.text(.12+.25*i,.63,a,ha='center',fontsize=14,color=COLORS[i],weight='bold');ax.text(.12+.25*i,.31,b,ha='center',linespacing=1.7)
    ax.text(.5,.02,'同一seed内成对；训练噪声和设计一同随seed改变。没有算法随机初始化。',ha='center',fontsize=10,color='#475569');finish(fig,out,'06-experiment-design.png')
    fig,axs=plt.subplots(1,3,figsize=(11.4,4.1))
    for ax,sigma,c in zip(axs,[0,.2,.8],COLORS):
        curve(ax,sigma,0,'clean_risk','12次中位数',c,True);ax.axvline(24,color='black',ls='--',lw=1);ax.set(xscale='log',yscale='log',xlabel='参数 p（对数轴）',ylabel='干净目标风险（对数轴）',title=f'训练噪声 σ={sigma}');ax.set_xticks([3,12,24,48,96],[3,12,24,48,96]);ax.grid(alpha=.15);ax.legend(fontsize=9)
    fig.suptitle('细线是全部12次；仅显示时把小于1e-12的舍入误差放在1e-12',fontsize=11);finish(fig,out,'07-capacity-noise.png')
    fig,axs=plt.subplots(1,2,figsize=(10.5,4))
    for lam,c in zip([0,.01,.1],COLORS):
        curve(axs[0],.8,lam,'clean_risk',f'λ={lam}',c);curve(axs[1],.8,lam,'train_mse',f'λ={lam}',c)
    for ax in axs:ax.axvline(24,color='black',ls='--',lw=1);ax.set(xlabel='p',yscale='log');ax.legend()
    axs[0].set(title='固定σ=0.8：正则化改变风险曲线',ylabel='12次干净风险中位数');axs[1].set(title='惩罚不再精确插值',ylabel='12次训练MSE中位数（显示下限1e-12）');finish(fig,out,'08-ridge-control.png')
    fig,axs=plt.subplots(1,2,figsize=(10.5,4));sel=[r for r in rows if r['sigma']==.8 and r['ridge']==0]
    sc=axs[0].scatter([r['smallest_singular'] for r in sel],[r['clean_risk'] for r in sel],c=[r['p'] for r in sel],cmap='viridis',s=14);axs[0].set(xscale='log',yscale='log',xlabel='最小非零奇异值',ylabel='干净风险',title='谱病态与风险：有关系，不能只看一个数');fig.colorbar(sc,ax=axs[0],label='p')
    for p in [3,24,32,96]:
        rr=[r for r in sel if r['p']==p];axs[1].scatter([r['noise_amplification'] for r in rr],[r['realized_noise_norm2']/.64 for r in rr],label=f'p={p}')
    axs[1].set(xscale='log',yscale='log',xlabel='噪声平均放大 Σ1/s²',ylabel='一次实际噪声范数² / σ²',title='一次噪声不等于对噪声取期望');axs[1].legend(fontsize=9);finish(fig,out,'09-spectrum-risk.png')
    fig,axs=plt.subplots(1,2,figsize=(10.5,4));c=d['conditional-risk'];axs[0].hist(c['noise_risks'],bins=28,color=COLORS[0],alpha=.8);axs[0].axvline(c['noise_expected_risk'],color=COLORS[1],lw=2,label='条件期望');axs[0].set(xlabel='固定X，重新抽噪声的干净风险',ylabel='次数',title='400次噪声重复（不是400个独立设计）');axs[0].legend()
    labels=['训练数据也随机\n总体理论期望','固定完整训练X\n平均训练噪声','固定X和训练y\n只平均新输入'];axs[1].axis('off')
    for i,txt in enumerate(labels):axs[1].text(.5,.86-i*.32,txt,ha='center',va='center',fontsize=13,bbox={'boxstyle':'round,pad=.6','fc':'#eff6ff','ec':COLORS[i]})
    finish(fig,out,'10-risk-expectations.png')
    fig,ax=plt.subplots(figsize=(10.5,4));ax.axis('off');titles=['存在性','算法选择','概率界','比例渐近'];top=['紧集连续目标\n连续sigmoid\n可用任意足够宽度','线性平方损失\n零初始化GD\n稳定步长与收敛','固定有限函数类\n独立同分布样本\n损失有界','n和p一起趋于无穷\n比例γ≠1\n谱及噪声条件'];bottom=['不提供训练方法','不覆盖任意深网','不是现实分数预测','不是n=24精确等式']
    for i in range(4):
        ax.text(.12+i*.25,.9,titles[i],ha='center',color=COLORS[i],weight='bold',fontsize=15);ax.text(.12+i*.25,.53,top[i],ha='center',linespacing=1.6);ax.text(.12+i*.25,.10,bottom[i],ha='center',fontsize=10,color='#9a3412')
    finish(fig,out,'11-theory-boundaries.png')
    fig,ax=plt.subplots(figsize=(8.8,4));ns=np.arange(5,1001)
    for m,c in zip([1,100,1000000],COLORS):ax.plot(ns,np.sqrt(np.log(2*m/.05)/(2*ns)),label=f'M={m}',color=c)
    ax.axhline(1,color='black',ls='--',label='风险差的平凡上界1');ax.axvline(24,color='#64748b',ls=':');ax.set(xscale='log',xlabel='独立样本数 n',ylabel='统一偏差上界 ε',title='δ=0.05；有界损失，有限且预先固定的M个候选');ax.legend();finish(fig,out,'12-finite-bound.png')
    print(json.dumps({'figures':12,'output':out.name},ensure_ascii=False))
if __name__=='__main__':main()
