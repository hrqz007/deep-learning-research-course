"""Regenerate every diagram from transparent operations and replay results."""
from pathlib import Path
import os,json,argparse
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'tmp/mpl'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(__file__).resolve().parent/'tmp/cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from matplotlib.patches import FancyBboxPatch
import torch
from experiment import sinusoidal,rotate_pairs
ROOT=Path(__file__).resolve().parent

def configure():
    path=Path(os.environ.get('DL_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
    if not path.exists():raise FileNotFoundError('Install Noto Sans CJK and set DL_CJK_FONT to its file')
    font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(path)).get_name(),
        'font.size':12,'axes.unicode_minus':False,'savefig.dpi':160,'axes.spines.top':False,'axes.spines.right':False})

def save(fig,path):
    fig.savefig(path,bbox_inches='tight',facecolor='white');plt.close(fig)

def boxes(ax,items):
    ax.set(xlim=(0,10),ylim=(0,10));ax.axis('off')
    for i,label in enumerate(items):
        y=9-i*1.65
        ax.add_patch(FancyBboxPatch((1,y-.6),8,1.1,boxstyle='round,pad=.12',fc='#e7f0f7',ec='#367699'))
        ax.text(5,y,label,ha='center',va='center',fontsize=11)
        if i<len(items)-1:ax.annotate('',(5,y-1.1),(5,y-.65),arrowprops={'arrowstyle':'->','color':'#367699'})

def main(results=ROOT/'outputs/results.json',output=ROOT/'figures'):
    configure();out=Path(output);out.mkdir(parents=True,exist_ok=True);r=json.loads(Path(results).read_text())
    fig,axes=plt.subplots(1,3,figsize=(13,5.4))
    for ax,title,items in zip(axes,['Encoder 层','Decoder-only 层','Encoder-decoder 的 decoder 层'],[
        ['源词元 + 位置','双向自注意力 + 残差 / LN','逐位置 FFN + 残差 / LN','输出 B × S × D'],
        ['已知目标前缀 + 位置','因果自注意力 + 残差 / LN','逐位置 FFN + 残差 / LN','输出 B × T × D'],
        ['右移目标 + 位置','因果自注意力 + 残差 / LN','交叉注意力 + 残差 / LN','逐位置 FFN + 残差 / LN','输出 B × T × D']]):
        boxes(ax,items);ax.set_title(title)
    fig.text(.79,.03,'交叉注意力：Q 来自目标，K/V 来自编码器',ha='center',fontsize=11)
    fig.subplots_adjust(wspace=.1,bottom=.09);save(fig,out/'01_architecture.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    source=np.arange(24).reshape(3,8)
    for ax,a,title in [(axes[0],source,'原张量 L=3，D=8'),(axes[1],source.reshape(3,2,4).transpose(1,0,2).reshape(6,4),'拆成 H=2，每头 d=4')]:
        ax.imshow(a,cmap='Blues');ax.set_title(title);ax.set_xlabel('特征坐标');ax.set_ylabel('位置 / 头内位置')
        for (i,j),v in np.ndenumerate(a):ax.text(j,i,str(v),ha='center',va='center',color='black' if v<16 else 'white')
    fig.tight_layout();save(fig,out/'02_heads.png')
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    boxes(axes[0],['X','H = X + Attn(LN1(X))','Y = H + FFN(LN2(H))','恒等路径保留在归一化外'])
    boxes(axes[1],['X','H = LN1(X + Attn(X))','Y = LN2(H + FFN(H))','相加之后仍需经过归一化'])
    axes[0].set_title('Pre-norm');axes[1].set_title('Post-norm');fig.tight_layout();save(fig,out/'03_norm.png')
    pe=sinusoidal(torch.arange(32),8).numpy();fig,axes=plt.subplots(1,2,figsize=(11,4))
    im=axes[0].imshow(pe.T,aspect='auto',cmap='coolwarm',vmin=-1,vmax=1);axes[0].set(xlabel='位置 p',ylabel='坐标',title='正弦位置向量 D=8');fig.colorbar(im,ax=axes[0],shrink=.7)
    for j in [0,2,4,6]:axes[1].plot(pe[:,j],label=f'sin 坐标 {j}')
    axes[1].set(xlabel='位置 p',ylabel='编码值',title='不同频率提供不同尺度');axes[1].legend(fontsize=10)
    fig.tight_layout();save(fig,out/'04_positions.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    angles=np.linspace(0,2*np.pi,200)
    for ax,p in zip(axes,[0,1]):
        ax.plot(np.cos(angles),np.sin(angles),c='#bbb');ax.axhline(0,c='#ddd');ax.axvline(0,c='#ddd')
        vec=[np.cos(p),np.sin(p)];ax.arrow(0,0,*vec,width=.012,head_width=.08,length_includes_head=True,color='#227899')
        ax.set(xlim=(-1.3,1.3),ylim=(-1.3,1.3),aspect='equal',xlabel='偶坐标',ylabel='奇坐标',title=f'向量 (1,0) 旋转 {p} rad')
    fig.tight_layout();save(fig,out/'05_rope.png')
    s=r['structure'];names=['无位置 / 同时换序','固定位置 / 只换词元','因果 / 修改未来','双向 / 修改未来','RoPE / 同移位置']
    vals=[s[k] for k in ['permutation_no_position_max','permutation_fixed_positions_max','causal_future_change_max','bidirectional_future_change_max','rope_joint_shift_max']]
    fig,ax=plt.subplots(figsize=(11,4));ax.barh(names,vals,color=['#658fa3','#d28a45','#658fa3','#d28a45','#658fa3']);ax.set(xlabel='最大绝对输出差（同一固定未训练块）',title='结构性零差与有意非零差都属于预期结果')
    for i,v in enumerate(vals):ax.text(v+.02,i,f'{v:.3g}',va='center')
    ax.set_xlim(0,max(vals)*1.23);fig.tight_layout();save(fig,out/'06_invariants.png')
    hand=r['hand'];fig,axes=plt.subplots(1,2,figsize=(11,3.8));a=np.array(hand[0]['a'])[0]
    axes[0].imshow(a,vmin=0,vmax=1,cmap='Blues');axes[0].set(xticks=[0,1],yticks=[0,1],xlabel='key',ylabel='query',title='手算例注意力')
    for (i,j),v in np.ndenumerate(a):axes[0].text(j,i,f'{v:.6f}',ha='center',va='center',color='white' if v>.5 else 'black')
    axes[1].plot([0,1,2],[h['loss'] for h in hand],marker='o');axes[1].set(xticks=[0,1,2],xlabel='已完成 SGD 更新数',ylabel='训练 half-MSE',title='同一实例的两次同步更新')
    fig.tight_layout();save(fig,out/'07_hand.png')
    print('Rendered',len(list(out.glob('*.png'))),'figures');return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',default=str(ROOT/'outputs/results.json'));p.add_argument('--output',default=str(ROOT/'figures'));a=p.parse_args();main(a.results,a.output)
