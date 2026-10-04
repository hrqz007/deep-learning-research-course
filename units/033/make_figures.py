"""Nine original teaching diagrams. Verify fixed inputs/results before any plotting."""
from pathlib import Path
import hashlib, json, io, os, tempfile
HERE=Path(__file__).resolve().parent

def check_fixed_artifacts():
    expected=json.loads((HERE/'data/figure-input-sha256.json').read_text())
    for name,sha in expected.items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=sha:raise ValueError('fixed figure input changed: '+name)
    return expected

def main():
    check_fixed_artifacts()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager,FontProperties
    from matplotlib.patches import FancyBboxPatch
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if not font.is_file():raise RuntimeError('Noto Sans CJK font required for rebuilding diagrams')
    fontManager.addfont(str(font));plt.rcParams.update({'font.family':FontProperties(fname=str(font)).get_name(),'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
    C=['#216A91','#2A8874','#C48327','#BA4D60'];gray='#64748b';pale='#EDF3F7'
    names=['SGD 固定','动量 固定','动量 余弦','动量 预热余弦'];keys=['sgd_fixed','momentum_fixed','momentum_cosine','momentum_warmup_cosine']
    Q=json.loads((HERE/'outputs/quadratic.json').read_text());N=json.loads((HERE/'outputs/network.json').read_text());T=json.loads((HERE/'outputs/trace.json').read_text());R=json.loads((HERE/'outputs/resume.json').read_text());blobs={}
    def save(fig,name):
        b=io.BytesIO();fig.savefig(b,format='png',bbox_inches='tight',facecolor='white');blobs[name]=b.getvalue();plt.close(fig)
    def box(ax,x,y,w,h,text,color=C[0],size=11):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=pale,edgecolor=color,lw=1.5));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
    def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.7,'color':gray})
    fig,ax=plt.subplots(figsize=(10.5,4.2));ax.axis('off');ax.set(xlim=(0,10.5),ylim=(0,4.2))
    for x,text,c in [(0.2,'当前参数 θ[k]\n同一批 X、y',C[0]),(2.9,'前向 → 损失\n反向 → 梯度 g[k]',C[0]),(5.6,'b[k] = μb[k-1] + g[k]\n保留方向历史',C[1]),(8.3,'新参数 = 旧参数\n− 本次学习率 × 缓冲',C[2])]:box(ax,x,2.1,2,1.15,text,c,10)
    for a,b in [(2.25,2.8),(4.95,5.5),(7.65,8.2)]:arrow(ax,(a,2.67),(b,2.67))
    ax.text(5.3,.95,'梯度告诉我们当前局部变化；动量改变方向历史；学习率控制本次位移。',ha='center',fontsize=12)
    ax.text(5.3,.25,'只有完成一次 optimizer.step()，本讲的 k 才增加 1。',ha='center',color=gray)
    save(fig,'01_update_mechanism.png')
    fig,ax=plt.subplots(figsize=(10.5,5.2));ax.axis('off');ax.set(xlim=(0,10.5),ylim=(0,5.2))
    for j,(label,vals) in enumerate([('输入 X','[1, 2]\n[−1, 1]'),('仿射 Z','[1.2, 0.8]\n[−0.1, 0.8]'),('ReLU H','[1.2, 0.8]\n[0, 0.8]'),('预测 P','0.42\n−0.30')]):
        box(ax,.25+j*2.65,3.05,2.0,1.25,label+'\n'+vals,C[0],10)
        if j<3:arrow(ax,(2.32+j*2.65,3.65),(2.83+j*2.65,3.65))
    box(ax,7.7,.65,2.55,1.25,'标签 [0.7, −0.2]\n残差 [−0.28, −0.10]\nL = 0.0221',C[2],10)
    box(ax,3.0,.65,3.9,1.25,'反向：残差 / 2 → 乘输出权重\n→ 乘 ReLU 门 → 乘输入并累加',C[1],10)
    arrow(ax,(9.25,2.95),(9.1,2.0));arrow(ax,(7.62,1.26),(7.0,1.26))
    ax.text(1.35,1.25,'第二行第一个\nReLU 门关闭',ha='center',color=C[3],fontsize=11)
    save(fig,'02_full_hand_graph.png')
    fig,ax=plt.subplots(figsize=(10.5,4.2));A=np.array(T['steps'][0]['per_sample_parameter_contribution']);x=np.arange(9)
    ax.bar(x-.2,A[0],width=.38,color=C[0],label='样本1贡献');ax.bar(x+.2,A[1],width=.38,color=C[2],label='样本2贡献');ax.plot(x,A.sum(axis=0),'o-',color=C[1],label='按参数逐列相加')
    ax.set_xticks(x,T['parameter_order']);ax.set_ylabel('对平均损失的梯度贡献');ax.axhline(0,color=gray,lw=.7);ax.legend(ncol=3,loc='lower left',fontsize=10);ax.set_ylim(-.255,.205);save(fig,'03_gradient_accumulation.png')
    fig,axs=plt.subplots(1,2,figsize=(10.5,4));lag=np.arange(20)
    for mu,c in [(.5,C[0]),(.8,C[1]),(.95,C[2])]:axs[0].plot(lag,(1-mu)*mu**lag,'o-',ms=3,label=f'μ={mu}',color=c)
    axs[0].set(xlabel='梯度的滞后步数 j',ylabel='归一化 EMA 权重 (1−μ)μ^j');axs[0].legend()
    g=np.r_[np.ones(12),-np.ones(18)];b=0.;m=[]
    for val in g:b=.8*b+.2*val;m.append(b)
    axs[1].step(np.arange(30),g,where='post',label='输入方向突变',color=C[2]);axs[1].plot(m,'o-',ms=3,label='μ=0.8 的 EMA',color=C[1]);axs[1].axhline(0,color=gray,lw=.7);axs[1].set(xlabel='更新索引',ylabel='梯度 / EMA');axs[1].legend(fontsize=10)
    fig.tight_layout();save(fig,'04_ema_memory.png')
    fig,axs=plt.subplots(1,2,figsize=(10.5,4.3));xx=np.linspace(-1.2,3.3,250);yy=np.linspace(-1.2,1.2,250);xx,yy=np.meshgrid(xx,yy)
    axs[0].contour(xx,yy,(xx**2+40*yy**2)/2,levels=[.1,.5,1,2,4,8,16,25],colors='#d0d9df',linewidths=.8)
    for mode,c,label in zip(keys[:2],C[:2],names[:2]):
        q=Q[mode];axs[0].plot([r['x'] for r in q],[r['y'] for r in q],'.-',lw=1.1,ms=3,color=c,label=label)
    axs[0].plot(0,0,'*',color=C[3],ms=12);axs[0].set(xlabel='缓方向 x  曲率1',ylabel='陡方向 y  曲率40');axs[0].legend(fontsize=10)
    for mode,c,label in zip(keys,C,names):axs[1].semilogy([r['update'] for r in Q[mode]],[r['loss'] for r in Q[mode]],color=c,label=label)
    axs[1].set(xlabel='已完成 optimizer 更新数',ylabel='二次函数值  对数轴');axs[1].legend(fontsize=9);fig.tight_layout();save(fig,'05_quadratic_valley.png')
    fig,ax=plt.subplots(figsize=(10.5,4));
    for mode,c,label in zip(keys[1:],C[1:],names[1:]):ax.plot(np.arange(72),[N[mode]['history'][k+1]['lr_used'] for k in range(72)],label=label,color=c,lw=2)
    ax.axvline(11,color=gray,ls=':',lw=1);ax.text(12,.064,'预热最后一次更新 k=11',fontsize=10,color=gray);ax.set(xlabel='本次 optimizer 更新索引 k  从0计数',ylabel='本次实际使用学习率 α[k]');ax.legend(loc='lower left',ncol=3,fontsize=10);ax.set_ylim(0,.069);save(fig,'06_learning_rate_schedules.png')
    fig,axs=plt.subplots(1,2,figsize=(10.5,4.2));axs[0].axis('off');axs[0].set(xlim=(0,5),ylim=(0,4.2))
    for x in [.1,2.6]:box(axs[0],x,2.55,2.1,.9,'4样本前向 + 反向',C[0],10)
    box(axs[0],1.0,.8,3.0,1,'8样本梯度聚合\n一次 optimizer + scheduler',C[1],10);arrow(axs[0],(1.15,2.45),(2.0,1.9));arrow(axs[0],(3.65,2.45),(3.,1.9));axs[0].text(2.5,.05,'48样本 / epoch → 12 microbatch → 6更新',ha='center',fontsize=9)
    import csv
    rows=list(csv.DictReader((HERE/'outputs/schedules.csv').open()))
    axs[1].plot([float(r['momentum_warmup_cosine']) for r in rows],color=C[1],label='正确：按更新推进')
    axs[1].plot([float(r['wrong_microbatch_clock']) for r in rows],color=C[3],label='错误示意：时钟加速2倍')
    axs[1].set(xlabel='真实 optimizer 更新索引',ylabel='学习率');axs[1].legend(fontsize=9);fig.tight_layout();save(fig,'07_accumulation_clock.png')
    fig,axs=plt.subplots(1,2,figsize=(10.5,4.2))
    for mode,c,label in zip(keys,C,names):
        h=N[mode]['history'];axs[0].semilogy([r['update'] for r in h],[r['loss'] for r in h],color=c,label=label)
    axs[0].set(xlabel='已完成更新数  四方案均72',ylabel='固定48点 half-MSE  对数轴');axs[0].legend(fontsize=9)
    vals=[N[m]['history'][-1]['loss'] for m in keys];axs[1].bar(np.arange(4),vals,color=C);axs[1].set_xticks(range(4),['SGD\n固定','动量\n固定','动量\n余弦','动量\n预热余弦']);axs[1].set_ylabel('第72次更新后固定数据损失')
    for i,v in enumerate(vals):axs[1].text(i,v+0.000025,f'{v:.6f}',ha='center',fontsize=9)
    axs[1].set_ylim(0,.0018);fig.tight_layout();save(fig,'08_network_comparison.png')
    fig,axs=plt.subplots(1,2,figsize=(10.5,4.2));axs[0].axis('off');axs[0].set(xlim=(0,5),ylim=(0,4.2))
    box(axs[0],.3,2.65,4.3,1.05,'epoch4结束：24次更新已完成\n下次应使用 α[24] = 0.05398294',C[0],10)
    box(axs[0],.3,.65,4.3,1.25,'参数 + 动量缓冲 + optimizer组\nscheduler计数 + 采样RNG + 原配置\n只继续剩余48次更新',C[1],10);arrow(axs[0],(2.45,2.5),(2.45,2.0))
    gaps=[0]+[R['negative_controls'][s]['max_parameter_gap'] for s in ['optimizer','scheduler','rng']]
    axs[1].bar(range(4),gaps,color=[C[1],C[2],C[3],C[0]]);axs[1].set_xticks(range(4),['完整恢复','漏动量','漏调度器','漏采样RNG']);axs[1].set_ylabel('最终参数最大绝对差')
    for i,v in enumerate(gaps):axs[1].text(i,v+.0004,f'{v:.5f}',ha='center',fontsize=9)
    axs[1].set_ylim(0,.022);fig.tight_layout();save(fig,'09_resume_state.png')
    # All plots were computed/serialized in memory. Replace each file atomically.
    target=HERE/'figures'
    if target.is_symlink():raise ValueError('figure output must not be a symlink')
    if any((target/name).is_symlink() for name in blobs):raise ValueError('unsafe figure target')
    target.mkdir(exist_ok=True)
    for name,blob in blobs.items():
        fd,tmp=tempfile.mkstemp(prefix='.033-',dir=target)
        try:
            with os.fdopen(fd,'wb') as f:f.write(blob)
            os.replace(tmp,target/name)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
    print(json.dumps({'figures':sorted(blobs),'count':len(blobs)},ensure_ascii=False))
if __name__=='__main__':main()
