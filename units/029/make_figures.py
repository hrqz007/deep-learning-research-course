"""Build original explanatory figures from guarded fixed input and report files."""
from pathlib import Path
import csv
import hashlib
import json

HERE=Path(__file__).resolve().parent

def check_fixed_artifacts():
    guard=json.loads((HERE/'data/figure-input-sha256.json').read_text())
    for name,digest in guard.items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('fixed figure input changed: '+name)
    return guard


def main():
    check_fixed_artifacts()  # before matplotlib import or any figure write
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager, FontProperties
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if not font.is_file():raise RuntimeError('Install Noto Sans CJK to rebuild Chinese figures')
    fontManager.addfont(str(font))
    from matplotlib.patches import FancyBboxPatch
    import numpy as np
    plt.rcParams.update({'font.family':FontProperties(fname=str(font)).get_name(),'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'savefig.dpi':180})
    colors={'blue':'#27658C','green':'#23836E','orange':'#D28531','red':'#B95254','gray':'#8997A6','pale':'#EAF1F5'}
    dest=HERE/'figures';dest.mkdir(exist_ok=True)
    def save(fig,name):fig.savefig(dest/name,bbox_inches='tight',facecolor='white');plt.close(fig)
    def panel(figsize=(10,3.5)):
        fig,ax=plt.subplots(figsize=figsize);ax.set_xlim(0,10);ax.set_ylim(0,3.5);ax.axis('off');return fig,ax
    def box(ax,x,y,w,h,text,c='blue'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=colors['pale'],edgecolor=colors[c],linewidth=1.5));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
    def arrow(ax,start,end,text=None):
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':colors['gray'],'lw':1.7})
        if text:ax.text((start[0]+end[0])/2,(start[1]+end[1])/2+.16,text,ha='center',fontsize=10)
    fig,ax=panel()
    box(ax,.25,1.2,2.2,1.2,'Python 名字 x\n指向张量对象');box(ax,3.7,2.2,2.6,.85,'shape=(2,2)\n索引两条样本');box(ax,3.7,.25,2.6,1.05,'float64 数值存储\n真实位置：CPU');box(ax,7.5,1.2,2.2,1.2,'计算来源\nleaf / grad_fn','green');arrow(ax,(2.5,1.8),(3.6,2.5));arrow(ax,(2.5,1.7),(3.6,.8));arrow(ax,(6.4,2.5),(7.45,2));arrow(ax,(6.4,.8),(7.45,1.5));ax.text(5,3.38,'数值、对象、设备、求导历史分别检查',ha='center',fontsize=14,weight='bold');save(fig,'01_tensor_planes.png')
    fig,ax=plt.subplots(figsize=(9.5,3.6));ax.axis('off')
    table=ax.table(cellText=[['h（直接引用）','共享','保留','原对象'],['h.clone()','独立','保留','非叶'],['h.detach()','共享','切断','不需梯度'],['h.detach().clone()','独立','切断','不需梯度'],['torch.tensor(h, requires_grad=True)','独立','切断','新叶']],colLabels=['表达式','数值存储','回到原 h 的路径','结果状态'],cellLoc='center',colWidths=[.46,.14,.20,.20],loc='center');table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2.05)
    for (i,j),cell in table.get_celld().items():cell.set_edgecolor('#D9E2EA');cell.set_facecolor('#DCE8F0' if i==0 else ('#F4F7F9' if i%2 else 'white'))
    ax.set_title('假设 h = w*w，w 是需要梯度的叶节点',pad=12);save(fig,'02_copy_history.png')
    fig,ax=panel();box(ax,.4,1.45,2,1,'w = 2\n叶，grad=12');box(ax,4,1.45,2,1,'h = w*w = 4\n非叶，保留 grad=3');box(ax,7.6,1.45,2,1,'L = 3*h = 12\n标量输出');arrow(ax,(2.5,2),(3.85,2),'乘法');arrow(ax,(6.15,2),(7.45,2),'乘3');arrow(ax,(7.4,.75),(6.1,.75),'上游 3');arrow(ax,(3.9,.75),(2.5,.75),'3 × (2w)');ax.text(5,3.1,'没有 retain_grad 时，h.grad 默认不保存；路径仍存在',ha='center',color=colors['green']);save(fig,'03_leaf_grad.png')
    rows=list(csv.DictReader((HERE/'outputs/comparisons.csv').open()));names=[];errors=[]
    for row in rows:
        if row['tensor'] not in names:names.append(row['tensor']);errors.append(0.)
        k=names.index(row['tensor']);errors[k]=max(errors[k],float(row['absolute_error']))
    fig,ax=plt.subplots(figsize=(10.5,3.8));ax.bar(range(len(names)),[max(x,1e-18) for x in errors],color=[colors['orange'] if n.startswith('d') else colors['blue'] for n in names]);ax.set_yscale('log');ax.set_ylim(5e-19,1e-12);ax.set_xticks(range(len(names)),names,rotation=55,ha='right');ax.set_ylabel('最大绝对误差');ax.set_title('同一网络逐张量对照：前向（蓝）与反向（橙）');ax.text(.01,.95,'严格为0的误差画在1e-18，仅用于显示',transform=ax.transAxes,va='top',fontsize=9);ax.grid(axis='y',alpha=.2);save(fig,'04_layer_comparison.png')
    probes=json.loads((HERE/'outputs/semantic-probes.json').read_text());a=probes['accumulation'];fig,ax=plt.subplots(figsize=(9,3.7));vals=[a['first'],a['uncleared_second'],a['after_clear']];bars=ax.bar(['第一次新前向+反向','第二次，不清 grad','清 grad 后，第三次'],vals,color=[colors['blue'],colors['red'],colors['green']],width=.6);ax.bar_label(bars,fmt='%g',padding=4);ax.set_ylim(0,10);ax.set_ylabel('w.grad');ax.set_title('w = 2 保持不变，每次对 w² 求导得到4');ax.grid(axis='y',alpha=.18);save(fig,'05_accumulation.png')
    fig,ax=plt.subplots(figsize=(9.6,3.3));columns=['dH0','dW1','db1','dZ1','dH1','dW2','db2','dW3','db3'];matrix=np.ones((3,len(columns)))
    for i,cut in enumerate(('detach','reconstruct'),1):
        for j,key in enumerate(columns):matrix[i,j]=float(key not in probes['cuts'][cut]['missing'])
    from matplotlib.colors import ListedColormap
    ax.imshow(matrix,cmap=ListedColormap(['#E3E8EC',colors['green']]),vmin=0,vmax=1,aspect='auto');ax.set_xticks(range(len(columns)),columns);ax.set_yticks(range(3),['完整图','H1.detach()','重建需梯度 H1']);ax.tick_params(length=0)
    for i in range(3):
        for j in range(len(columns)):ax.text(j,i,'有' if matrix[i,j] else 'None',ha='center',va='center',color='white' if matrix[i,j] else '#586575')
    ax.set_title('loss相同、末层梯度相同，仍可能已经断图',pad=14);save(fig,'06_cut_diagnosis.png')
    fig,ax=panel((10.5,3.7));box(ax,.2,1.6,2.55,1.1,'① 前向\nh = tanh(w)\nL = sum(h*h)');box(ax,3.7,1.6,2.55,1.1,'② 原地改写\nh.add_(1)','orange');box(ax,7.15,1.6,2.55,1.1,'③ backward\n保存版本不匹配\nRuntimeError','red');arrow(ax,(2.9,2.15),(3.55,2.15));arrow(ax,(6.4,2.15),(7,2.15));ax.text(1.47,.83,'原来保存的 h',ha='center',color=colors['blue']);ax.text(4.97,.83,'同一存储，新数值',ha='center',color=colors['orange']);ax.text(8.42,.83,'不能拿新值解释旧loss',ha='center',color=colors['red']);ax.text(5,3.25,'时间先后是错误的一部分',ha='center',fontsize=14);save(fig,'07_inplace_timeline.png')
    fig,ax=plt.subplots(figsize=(9.6,3.8));ax.axis('off');table=ax.table(cellText=[['train()','[0,0]','[0,0]，不记录新图'],['eval()','[1,2]，梯度[1,1]','[1,2]，不记录新图']],colLabels=['Dropout(p=1)','普通梯度模式','no_grad'],cellLoc='center',colWidths=[.24,.36,.40],loc='center');table.auto_set_font_size(False);table.set_fontsize(11);table.scale(1,3)
    for (i,j),cell in table.get_celld().items():cell.set_edgecolor('#D9E2EA');cell.set_facecolor('#DCE8F0' if i==0 else '#F4F7F9')
    ax.set_title('确定性探针：输出取 module(x)+0，x=[1,2]',pad=4);ax.text(.5,.03,'加0保证新运算发生；eval直接返回已有输入时，其原requires_grad标志仍在。',transform=ax.transAxes,ha='center',fontsize=10);save(fig,'08_modes.png')
    print(json.dumps({'figures':len(list(dest.glob('*.png'))),'input_guard_files':len(check_fixed_artifacts())}))

if __name__=='__main__':main()
