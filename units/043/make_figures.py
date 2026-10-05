"""Regenerate all original, data-driven teaching figures."""
from pathlib import Path
import os,argparse,json
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'../../tmp/043-mpl'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'../../tmp/043-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle,FancyArrowPatch
import numpy as np
FONT=os.environ.get('DL_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if not Path(FONT).is_file():raise FileNotFoundError('Install Noto Sans CJK or set DL_CJK_FONT to a CJK font file')
plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'font.size':11,'axes.unicode_minus':False,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
BLUE='#2468a2';ORANGE='#d67d20';GREEN='#28836c';RED='#bc4141'
def val(xs):return np.array([x['float'] for x in xs])
def grid(ax,m,title,limits=None,fmt='.3g',cmap='RdBu_r'):
    m=np.array(m);v=max(1e-12,float(np.abs(m).max())) if limits is None else limits
    ax.imshow(m,cmap=cmap,vmin=-v,vmax=v);ax.set_title(title,pad=10);ax.set_xticks(range(m.shape[1]));ax.set_yticks(range(m.shape[0]));ax.set_xlabel('列 c');ax.set_ylabel('行 r')
    if m.size<=36:
        for ij in np.ndindex(m.shape):ax.text(ij[1],ij[0],format(m[ij],fmt),ha='center',va='center',color='white' if abs(m[ij])>.7*v else '#14202b',fontsize=10)
def main(out):
    out.mkdir(parents=True,exist_ok=True);d=json.loads((ROOT/'data/experiment_data.json').read_text());r=json.loads((ROOT/'outputs/results.json').read_text());h=r['hand'];S=r['symmetry'];L=r['learning']
    def save(fig,name):fig.savefig(out/name,bbox_inches='tight');plt.close(fig)
    xs=[[[1,0,2],[0,1,0],[2,0,1]],[[0,1,0],[1,0,1],[0,1,0]]]
    fig,ax=plt.subplots(1,3,figsize=(10,3));grid(ax[0],xs[0],'输入 X1：3 × 3');ax[0].add_patch(Rectangle((-.48,-.48),1.96,1.96,fill=False,ec=ORANGE,lw=3));grid(ax[1],[[.5,-.25],[.25,.5]],'共享核 K：2 × 2');grid(ax[2],val(h[0]['samples'][0]['z']).reshape(2,2),'Z1：逐窗乘加，再加 b = 0.1');fig.tight_layout();save(fig,'01_window.png')
    fig,ax=plt.subplots(1,2,figsize=(10,3.5));A=np.zeros((4,9));K=np.array([[1,2],[3,4]])
    for n in range(4):
        rr,cc=divmod(n,2)
        for a,b in np.ndindex((2,2)):A[n,(rr+a)*3+cc+b]=K[a,b]
    ax[0].imshow(A,cmap='viridis',vmin=0,vmax=4);ax[0].set_xticks(range(9));ax[0].set_yticks(range(4));ax[0].set_xlabel('展平输入坐标');ax[0].set_ylabel('展平输出坐标');ax[0].set_title('稀疏连接：同色位置使用同一个核参数')
    for ij in np.ndindex(A.shape):ax[0].text(ij[1],ij[0],'.' if A[ij]==0 else f'k{int(A[ij])}',ha='center',va='center',color='white' if A[ij]<3 else 'black')
    ax[1].bar(['全连接\n9→4','局部不共享\n4个2×2窗','共享卷积\n2×2核'],[40,20,5],color=[BLUE,ORANGE,GREEN]);ax[1].set_ylabel('可学习参数（含偏置）');ax[1].set_title('同样4个输出，约束方式不同')
    for i,v in enumerate([40,20,5]):ax[1].text(i,v+1,str(v),ha='center');
    fig.tight_layout();save(fig,'02_sharing.png')
    fig,ax=plt.subplots(1,3,figsize=(10,3.2))
    for a,(s,di,p,title) in zip(ax,[(1,1,0,'k=3，s=1，d=1，p=0'),(2,1,0,'k=3，s=2，d=1，p=0'),(1,2,0,'k=3，s=1，d=2，p=0')]):
        a.set_xlim(-.6,6.6);a.set_ylim(2,-1);a.set_yticks([]);a.set_xticks(range(7));a.set_xlabel('一条长度7的输入轴');a.set_title(title)
        for j in range(7):a.add_patch(Rectangle((j-.35,-.25),.7,.5,facecolor='#e5ebf1',edgecolor='white'))
        count=(7-di*2-1)//s+1
        for j in range(3):a.plot(di*j,0,'o',color=ORANGE,ms=10)
        if count>1:
            for j in range(3):a.plot(s+di*j,1,'o',color=BLUE,ms=8)
        a.text(3,1.7,f'有效核宽 {di*2+1}；输出 {count}',ha='center')
    fig.tight_layout();save(fig,'03_geometry.png')
    fig,ax=plt.subplots(2,3,figsize=(10,5))
    for i in range(2):
        grid(ax[i,0],xs[i],f'样本{i+1}，目标 {(.5,1)[i]}');grid(ax[i,1],val(h[0]['samples'][i]['z']).reshape(2,2),'卷积 Z');grid(ax[i,2],val(h[0]['samples'][i]['h']).reshape(2,2),f'ReLU H，均值 {h[0]["samples"][i]["pred"]["float"]:.4f}')
    fig.tight_layout();save(fig,'04_hand_forward.png')
    fig,ax=plt.subplots(figsize=(9,3));a=np.array([val(s['contribution']) for s in h[0]['samples']]+[val(h[0]['gradient'])]);v=np.abs(a).max();ax.imshow(a,cmap='RdBu_r',vmin=-v,vmax=v,aspect='auto');ax.set_xticks(range(5),['k00','k01','k10','k11','b']);ax.set_yticks(range(3),['样本1（已含1/2）','样本2（已含1/2）','相加：全批次梯度']);ax.set_title('先对位置相加，再对样本相加；偏置也共享')
    for ij in np.ndindex(a.shape):ax.text(ij[1],ij[0],f'{a[ij]:.6f}',ha='center',va='center',color='white' if abs(a[ij])>.65*v else 'black')
    fig.tight_layout();save(fig,'05_gradient.png')
    fig,ax=plt.subplots(1,2,figsize=(9,3.2))
    for i in range(2):grid(ax[i],[[v['float'] for v in row] for row in h[0]['samples'][i]['dx']],f'样本{i+1}：输入梯度 dL/dX',limits=.05,fmt='.4f')
    fig.tight_layout();save(fig,'06_input_gradient.png')
    fig,ax=plt.subplots(1,2,figsize=(10,3.2));x=np.arange(3);ax[0].plot(x,[s['loss']['float'] for s in h],'o-',label='平均损失',color=BLUE)
    for i,c in enumerate([ORANGE,GREEN]):ax[0].plot(x,[s['samples'][i]['loss']['float'] for s in h],'s--',label=f'样本{i+1}损失',color=c)
    ax[0].set_xticks(x,['初值','1次更新','2次更新']);ax[0].set_ylabel('half-MSE');ax[0].legend();ax[0].set_title('总损失下降，不保证每个样本改善')
    # Dependency sets for a 3x3 dilation-2 layer on top of a 3x3 stride-2 layer.
    supports=[set(range(3)),set(a*2+b for a in [0,2,4] for b in range(3))]
    for row,sp in enumerate(supports):
        for t in range(11):ax[1].plot(t,row,'o',color=BLUE if t in sp else '#cbd5e1',ms=9)
        ax[1].plot([min(sp),max(sp)],[row+.2,row+.2],color=ORANGE,lw=2)
    ax[1].set_yticks([0,1],['第一层：k3 s2','第二层：k3 d2 s1']);ax[1].set_xticks(range(11));ax[1].set_ylim(-.6,1.7);ax[1].set_xlabel('输入位置');ax[1].set_title('感受野包络宽11，但位置3和7未被使用')
    fig.tight_layout();save(fig,'07_updates_rf.png')
    fig,ax=plt.subplots(2,3,figsize=(10,5.4))
    for i,name in enumerate(['zero_stride1_full','circular_stride1']):
        a=S['arrays'][name];limit=max(np.abs(a['left']).max(),np.abs(a['right']).max())
        for j,(key,title) in enumerate([('left','先平移，再卷积'),('right','先卷积，再平移'),('difference','绝对差')]):
            mat=np.array(a[key]);ax[i,j].imshow(mat,cmap='magma' if j==2 else 'RdBu_r',vmin=0 if j==2 else -limit,vmax=4.5 if j==2 else limit);ax[i,j].set_title(('零边界' if i==0 else '周期边界')+'：'+title);ax[i,j].set_xticks([0,7]);ax[i,j].set_yticks([0,7])
            if j==2:ax[i,j].text(3.5,8.8,f'最大差 {mat.max():.3g}',ha='center')
    fig.tight_layout();save(fig,'08_boundary.png')
    fig,ax=plt.subplots(1,3,figsize=(10,3.2));names=['circular_stride2_shift2','circular_stride2_shift1_vs_shift0','circular_stride2_shift1_vs_shift1'];titles=['移2输入格 ↔ 移1输出格','移1输入格，硬比输出不移','移1输入格，硬比输出移1格']
    for a,name,title in zip(ax,names,titles):
        m=np.array(S['arrays'][name]['difference']);a.imshow(m,cmap='magma',vmin=0,vmax=14.5);a.set_title(title,fontsize=10);a.set_xticks(range(4));a.set_yticks(range(4));a.set_xlabel(f'最大差 {m.max():.3g}')
    fig.suptitle('步幅2：单像素平移没有整数输出格对应，后两项只是错误对齐的诊断');fig.tight_layout();save(fig,'09_stride.png')
    fig,ax=plt.subplots(1,3,figsize=(10,3.2));tr=L['trace'];ax[0].semilogy([v['step'] for v in tr],[v['loss'] for v in tr],color=BLUE,label='固定80步SGD');ax[0].axhline(L['least_squares_train_half_mse'],color=ORANGE,ls='--',label='最小二乘训练参照');ax[0].set_xlabel('参数更新次数');ax[0].set_ylabel('训练 half-MSE');ax[0].legend(fontsize=9);ax[0].set_title('仍有优化误差')
    grid(ax[1],L['true_kernel'],'生成核（用于教学核验）',limits=.5);grid(ax[2],L['learned_kernel'],'80步学习核',limits=.5,fmt='.2f');fig.tight_layout();save(fig,'10_learning.png')
    fig,ax=plt.subplots(1,2,figsize=(10,3.4));label=['周期 + GAP','零边界 + GAP','周期 + 位置加权'];before=[S['heads'][k][0] for k in ['circular_gap','zero_gap','circular_position_weighted']];after=[S['heads'][k][1] for k in ['circular_gap','zero_gap','circular_position_weighted']]
    ax[0].bar(range(3),np.abs(np.array(after)-before),color=[GREEN,ORANGE,RED]);ax[0].set_xticks(range(3),label,rotation=10);ax[0].set_ylabel('右移1像素后 |标量变化|');ax[0].set_title('不变性还取决于边界和任务头')
    per=((np.array(L['test_prediction'])-np.array(L['test_target']))**2).mean((1,2,3))/2;ax[1].bar(range(1,21),per,color=BLUE);ax[1].axhline(L['test_half_mse'],color=ORANGE,ls='--',label='20图均值');ax[1].set_xlabel('测试图像编号');ax[1].set_ylabel('每图 half-MSE');ax[1].set_title('保留全部20幅测试图的误差');ax[1].legend();fig.tight_layout();save(fig,'11_heads_test.png')
    print(json.dumps({'figures':len(list(out.glob('*.png'))),'font':FONT},ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'figures');main(p.parse_args().output)
