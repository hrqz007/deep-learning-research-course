"""Original diagrams. Fixed-input/report guards run before plotting or writes."""
from pathlib import Path
import csv
import hashlib
import json
HERE=Path(__file__).resolve().parent


def check_fixed_artifacts():
    expected=json.loads((HERE/'data/figure-input-sha256.json').read_text())
    for name,digest in expected.items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('fixed figure input changed: '+name)
    return expected


def main():
    check_fixed_artifacts()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.font_manager import fontManager,FontProperties
    from matplotlib.patches import FancyBboxPatch
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if not font.is_file():raise RuntimeError('Noto Sans CJK font needed to rebuild figures')
    fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':FontProperties(fname=str(font)).get_name(),'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':175})
    colors={'blue':'#24658F','green':'#25856C','orange':'#C68027','red':'#B84854','pale':'#EDF3F7','gray':'#657889'}
    (HERE/'figures').mkdir(exist_ok=True)
    def save(fig,name):fig.savefig(HERE/'figures'/name,bbox_inches='tight',facecolor='white');plt.close(fig)
    def box(ax,x,y,w,h,text,color='blue',size=10):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.07',facecolor=colors['pale'],edgecolor=colors[color],linewidth=1.4));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
    def arrow(ax,a,b,label=''):
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':colors['gray'],'lw':1.5})
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.09,label,ha='center',fontsize=9,color=colors['gray'])
    fig,ax=plt.subplots(figsize=(10.5,5.3));ax.set_xlim(0,10.5);ax.set_ylim(0,5.3);ax.axis('off')
    labels=['数据：样本 ID、配对、shape、范围','损失：手算一例、归约与基线','梯度：None / 非有限 / 数值核验','更新：参数身份、学习率、实际变化','单批次：固定数据能否拟合？']
    hints=['错 → 查加载与标签连接','错 → 查目标函数与广播','错 → 查第一处断图或算子','错 → 查优化器与调用时序','成功 → 再做扰动与独立评价']
    for i,(label,hint) in enumerate(zip(labels,hints)):
        y=4.35-i
        box(ax,.3,y,6.25,.62,label,size=11);box(ax,7.1,y,3.1,.62,hint,'orange' if i<4 else 'green',10)
        arrow(ax,(6.63,y+.31),(7.02,y+.31))
        if i<4:arrow(ax,(3.425,y-.02),(3.425,y-.3))
    ax.set_title('先固定一次失败，再按依赖关系排除',pad=10);save(fig,'01_debug_tree.png')
    fig,axes=plt.subplots(1,2,figsize=(9.6,3.5))
    x=np.linspace(-1.5,1.5,150);axes[0].plot(x,.5*x+.25,color=colors['blue'],label='初始预测');axes[0].scatter([-1,1],[-1,1],color=colors['orange'],label='目标');axes[0].set_xlabel('x');axes[0].set_ylabel('y');axes[0].legend();axes[0].set_title('手算样本：残差 0.75、-0.25')
    for xi,yi in [(-1,-1),(1,1)]:axes[0].plot([xi,xi],[yi,.5*xi+.25],':',color=colors['red'])
    axes[1].axis('off');box(axes[1],.08,.57,.84,.3,'L = 5/32 = 0.15625\ng = (-0.5, 0.25)',size=12);box(axes[1],.08,.08,.84,.3,'SGD：η = 0.1\n(w,b) → (0.55,0.225)\n新损失 = 81/640 = 0.1265625','green',11)
    fig.tight_layout();save(fig,'02_hand_step.png')
    fd=list(csv.DictReader((HERE/'outputs/finite-difference.csv').open()));steps=sorted({float(r['step']) for r in fd});errors=[max(float(r['abs_error']) for r in fd if float(r['step'])==h) for h in steps]
    fig,ax=plt.subplots(figsize=(9,3.8));ax.loglog(steps,errors,'o-',color=colors['blue']);ax.set_xlabel('有限差分步长 h');ax.set_ylabel('13坐标最大绝对误差');ax.set_title('太大的截断误差与太小的浮点相减误差');ax.grid(which='both',alpha=.2);ax.axvline(1e-6,color=colors['green'],linestyle=':',label='本例核验步长 1e-6');ax.legend();save(fig,'03_fd_steps.png')
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.7));x=np.linspace(-.8,.8,201);axes[0].plot(x,np.maximum(x,0),color=colors['blue']);axes[0].plot([-.6,.6],[0,.6],'--',color=colors['orange']);axes[0].scatter([0],[0],color=colors['red']);axes[0].set_title('ReLU在0：差分斜率0.5');axes[0].set_xlabel('x');axes[0].text(-.7,.68,'普通导数不存在\nautograd约定返回0',fontsize=10)
    ax=axes[1];ax.set_xlim(-.2,2.6);ax.set_ylim(-.2,2.6);ax.set_aspect('equal');ax.arrow(0,0,1,2,width=.02,color=colors['blue'],length_includes_head=True);ax.arrow(0,0,2,1,width=.02,color=colors['red'],length_includes_head=True);ax.plot([0,2.4],[0,2.4],':',color=colors['gray']);ax.text(.4,2.15,'g = (1,2)',color=colors['blue']);ax.text(1.6,.6,'错 g = (2,1)',color=colors['red']);ax.set_title('沿 (1,1) 的投影完全相同');ax.set_xlabel('坐标1');ax.set_ylabel('坐标2');fig.tight_layout();save(fig,'04_fd_limits.png')
    f=json.loads((HERE/'outputs/faults.json').read_text())['one_step'];modes=['healthy','no_step','zero_lr','omit_W1','detach_hidden','sum_reduction'];labels=['正常','没调step','lr=0','漏传W1','分离隐藏层','sum归约'];fig,axes=plt.subplots(1,2,figsize=(10.2,3.9));names=['W1','b1','W2','b2'];idx=np.arange(len(modes))
    for j,name in enumerate(names):
        vals=[f[m]['parameters'][j]['grad_norm'] for m in modes]
        axes[0].plot(idx,[np.nan if v is None else v for v in vals],marker='o',label=name)
        axes[1].plot(idx,[f[m]['parameters'][j]['update_norm'] for m in modes],marker='o',label=name)
    for ax in axes:ax.set_xticks(idx,labels,rotation=25,ha='right');ax.grid(alpha=.17)
    axes[0].set_title('每组参数的梯度范数');axes[0].set_ylabel('L2范数');axes[0].legend(ncol=4,fontsize=9);axes[0].text(.04,.80,'隐藏层断图：W1、b1为None\n曲线上留空，不能伪装成数值0',transform=axes[0].transAxes,fontsize=9)
    axes[1].set_title('同一次SGD后的参数变化');axes[1].set_ylabel('更新L2范数');fig.tight_layout();save(fig,'05_grad_update.png')
    rows=list(csv.DictReader((HERE/'outputs/training.csv').open()));fig,ax=plt.subplots(figsize=(9.4,3.8))
    for label,title,col in [('rule','规则标签',colors['blue']),('random','随机标签',colors['orange'])]:
        records=[r for r in rows if r['label_set']==label];ax.semilogy([int(r['step']) for r in records],[max(float(r['loss_before_step']),1e-32) for r in records],label=title,color=col)
    ax.axhline(1e-4,linestyle=':',color=colors['green'],label='本例诊断阈值1e-4');ax.set_xlabel('已经完成的更新次数');ax.set_ylabel('同一批次 half-MSE');ax.set_title('相同初始化与结构，两组标签都能记住');ax.legend();ax.grid(alpha=.2);ax.text(.02,.06,'仅图示把小于1e-32的值画到1e-32；原始数值保留在CSV',transform=ax.transAxes,fontsize=9);save(fig,'06_batch_fits.png')
    fig,ax=plt.subplots(figsize=(9,3.5));c=np.linspace(-1.6,1.6,250);ax.plot(c,(c*c+1)/2,color=colors['blue']);ax.axhline(.5,color=colors['orange'],linestyle='--');ax.scatter([0],[.5],color=colors['red']);ax.set_xlabel('同一个输入只能产生的预测 c');ax.set_ylabel('[(c+1)² + (c-1)²] / 4');ax.set_title('矛盾重复样本：任何确定性预测器都无法低于0.5');ax.grid(alpha=.2);save(fig,'07_duplicate_floor.png')
    p=json.loads((HERE/'outputs/perturbations.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,3.6));axes[0].bar(['原配对','只反转输入','输入置零'],[p['baseline_loss'],p['input_only_permutation_loss'],p['zero_input_loss_original_targets']],color=[colors['blue'],colors['red'],colors['orange']]);axes[0].set_ylabel('对原标签的half-MSE');axes[0].set_title('冻结模型后破坏输入关系')
    r=p['small_shifts'];axes[1].plot([v['epsilon'] for v in r],[v['mean_slope'] for v in r],'o-',color=colors['blue'],label='模型平均变化率');axes[1].axhline(.7,linestyle='--',color=colors['green'],label='真实规则变化率0.7');axes[1].set_xscale('log');axes[1].set_ylim(.55,.74);axes[1].set_xlabel('x1增加 ε');axes[1].set_title('原样本拟合极好，邻域机制仍不同');axes[1].legend(fontsize=9);fig.tight_layout();save(fig,'08_input_interventions.png')
    fig,ax=plt.subplots(figsize=(10,3.6));ax.set_xlim(0,10);ax.set_ylim(0,3.6);ax.axis('off')
    box(ax,.2,1.45,2.1,1.2,'现象\nW1没有变化');box(ax,2.8,1.45,2.45,1.2,'证据\n梯度=0.47057\n更新=0，未入优化器','orange');box(ax,5.75,1.45,1.8,1.2,'单变量修复\n传入W1','green');box(ax,8.05,1.45,1.7,1.2,'回归检查\n更新=0.01412','green')
    for a,b in [(2.4,2.7),(5.35,5.65),(7.65,7.95)]:arrow(ax,(a,2.05),(b,2.05))
    ax.text(5,.65,'证据指向“更新器漏参”；无需更换模型或损失',ha='center',fontsize=13);ax.set_title('可定位报告：把推断绑定到可复现的数字',pad=5);save(fig,'09_fault_report.png')


if __name__=='__main__':main()
