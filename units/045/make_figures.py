"""Original figures from frozen inputs, replay records and measured outputs."""
from pathlib import Path
import os,argparse,json,gzip
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'../../tmp/045-mpl'));os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'../../tmp/045-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
import numpy as np
import torch
from experiment import replay,resize_bilinear
FONT=os.environ.get('DL_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if not Path(FONT).is_file():raise FileNotFoundError('Install Noto Sans CJK or set DL_CJK_FONT')
plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'font.size':10,'axes.unicode_minus':False,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
COLORS=['#266b9f','#2d8b72','#c6584d','#9866af'];ARM_LABELS=['不增强','重采背景','重采+旋转旧标签','重采+旋转改标签']
def values(xs):return np.array([v['float'] for v in xs])
def tile(ax,m,title,low=0,high=1,annot=False,cmap='gray'):
    m=np.array(m);ax.imshow(m,cmap=cmap,vmin=low,vmax=high);ax.set_title(title);ax.set_xticks(range(m.shape[1]) if m.shape[1]<=4 else [0,m.shape[1]-1]);ax.set_yticks(range(m.shape[0]) if m.shape[0]<=4 else [0,m.shape[0]-1])
    if annot:
        for (r,c),v in np.ndenumerate(m):ax.text(c,r,f'{v:.3g}',ha='center',va='center',color=('white' if abs(v)>.65*max(abs(low),abs(high)) else 'black') if cmap=='RdBu_r' else ('white' if v<(low+high)/2 else 'black'))
def main(out):
    out.mkdir(parents=True,exist_ok=True);d=json.loads((ROOT/'data/images.json').read_text());p=json.loads((ROOT/'data/augmentation_plans.json').read_text());r=json.loads((ROOT/'outputs/results.json').read_text());g=r['geometry'];h=r['hand'];L=r['learning'];tr=json.loads(gzip.decompress((ROOT/'outputs/training_traces.json.gz').read_bytes()))
    def save(fig,n):fig.savefig(out/n,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(10,3));pixel=np.array([[[.9,.2],[.1,.5]],[[.1,.8],[.2,.4]],[[.2,.1],[.9,.6]]])
    for i,a in enumerate(ax):tile(a,pixel[i],f'通道 {i}：'+['R','G','B'][i],annot=True);a.set_xlabel('列 x 向右');a.set_ylabel('行 y 向下')
    fig.suptitle('同一个像素 (y=0,x=0) 的RGB值是(0.9,0.1,0.2)，通道不是批次');fig.tight_layout();save(fig,'01_channels.png')
    fig,ax=plt.subplots(1,3,figsize=(10,3));tile(ax[0],g['original_image'][0],'原图：边缘坐标框',annot=True);tile(ax[1],g['image'][0],'裁剪 + 水平翻转',annot=True);tile(ax[2],g['mask'],'掩码同样变换',0,2,True,'viridis')
    for a,boxes in [(ax[0],g['original_boxes']),(ax[1],g['boxes']),(ax[2],g['boxes'])]:
        for x0,y0,x1,y1 in boxes:a.add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,ec='#e36e24',lw=2))
    fig.suptitle('半开XYXY框：主框面积保留3/4，第二框完全裁掉；先裁剪，再翻转');fig.tight_layout();save(fig,'02_geometry.png')
    fig,ax=plt.subplots(1,3,figsize=(10,3));tile(ax[0],g['categorical_mask'],'原始类别ID：只有0与2',0,2,True,'viridis');tile(ax[1],g['wrong_bilinear_mask'],'错误：双线性产生类别1',0,2,True,'viridis');tile(ax[2],g['nearest_mask'],'最近邻：保持类别集合',0,2,True,'viridis');fig.tight_layout();save(fig,'03_mask_interpolation.png')
    fig,ax=plt.subplots(1,2,figsize=(10,3.2));arr=resize_bilinear(np.array([[[0.,1.],[2.,3.]]]),(3,3))[0];tile(ax[0],arr,'2×2 → 3×3：中心值1.5',0,3,True,'viridis');ax[1].set_xlim(-.7,1.7);ax[1].set_ylim(-.4,1.4);ax[1].set_yticks([0,1],['输入像素中心','输出反向映射']);ax[1].plot([0,1],[0,0],'o',ms=11,color=COLORS[0]);coords=[-1/6,.5,7/6];ax[1].plot(coords,[1]*3,'o',ms=9,color=COLORS[1]);ax[1].set_xticks(coords,['-1/6','1/2','7/6']);ax[1].set_title('半像素坐标：外侧夹到0和1');ax[1].set_xlabel('输入索引坐标');fig.tight_layout();save(fig,'04_half_pixel.png')
    raw=np.array([[[0,1],[.5,.25]],[[1,.5],[.75,.25]]]);fig,ax=plt.subplots(2,3,figsize=(9,5))
    for i in range(2):
        tile(ax[i,0],raw[i],f'原图{i+1}：目标均值 {raw[i].mean():.4f}',annot=True);tile(ax[i,1],raw[i,:,::-1],'固定水平翻转',annot=True);tile(ax[i,2],values(h[0]['samples'][i]['u']).reshape(2,2),'减0.5，除0.5',-1,1,True,'RdBu_r')
    fig.tight_layout();save(fig,'05_hand_transform.png')
    fig,ax=plt.subplots(1,2,figsize=(10,3.4));a=np.array([values(s['contribution']) for s in h[0]['samples']]+[values(h[0]['gradient'])]);v=np.abs(a).max();ax[0].imshow(a,cmap='RdBu_r',vmin=-v,vmax=v,aspect='auto');ax[0].set_xticks(range(5),['w0','w1','w2','w3','b']);ax[0].set_yticks(range(3),['样本1','样本2','合计']);ax[0].set_title('同一旧参数下的逐样本贡献')
    for ij in np.ndindex(a.shape):ax[0].text(ij[1],ij[0],f'{a[ij]:.4f}',ha='center',va='center',fontsize=9,color='white' if abs(a[ij])>.7*v else 'black')
    for i,color in enumerate(COLORS[:2]):ax[1].plot(range(3),[s['samples'][i]['loss']['float'] for s in h],'o-',color=color,label=f'样本{i+1}')
    ax[1].plot(range(3),[s['loss']['float'] for s in h],'s--',color=COLORS[2],label='两例平均');ax[1].set(xlabel='更新次数',ylabel='half-MSE',xticks=[0,1,2]);ax[1].legend();fig.tight_layout();save(fig,'06_hand_gradient.png')
    fig,ax=plt.subplots(2,3,figsize=(9,5));split=d['test']
    for i in range(2):
        for j,(view,title) in enumerate([('x','原相关背景'),('swapped','反相关背景'),('neutral','中性背景')]):tile(ax[i,j],split[view][i][0],f'{["竖条 y=0","横条 y=1"][i]}：{title}')
    fig.suptitle('同一测试图像的形状、位置和噪声保持相同；只重渲染背景');fig.tight_layout();save(fig,'07_background_views.png')
    # Select a predeclared rotated index (first flag=1) for a transparent replay example.
    idx=int(np.flatnonzero(p['4511']['rotate90'][0])[0]);fig,ax=plt.subplots(1,4,figsize=(11,3))
    for a,arm,label in zip(ax,L['protocol']['arms'],ARM_LABELS):
        image,label_y=replay(d['train'],p['4511'],0,arm);tile(a,image[idx,0],label+f'\ny={label_y[idx]}')
    fig.suptitle(f'固定回放：seed4511，第0步，样本{idx}；新背景 {p["4511"]["background"][0][idx]:.4f}，旋转=1');fig.tight_layout();save(fig,'08_replay.png')
    fig,ax=plt.subplots(1,2,figsize=(10,3.7));summary=L['summary'];xx=np.arange(4);width=.24
    for j,(view,label) in enumerate([('test_x','原相关'),('test_swapped','反相关'),('test_neutral','中性')]):
        ax[0].bar(xx+(j-1)*width,[summary[a][view]['nll'] for a in L['protocol']['arms']],width,label=label);ax[1].bar(xx+(j-1)*width,[summary[a][view]['accuracy'] for a in L['protocol']['arms']],width,label=label)
    for a in ax:a.set_xticks(xx,['无增强','背景','旋转旧标签','旋转改标签'],rotation=12)
    handles,labels=ax[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.02),ncol=3,frameon=False)
    ax[0].set_yscale('log');ax[0].set_ylabel('测试NLL，三训练种子均值');ax[0].set_title('原相关高分掩盖背景失败');ax[1].set_ylim(0,1.05);ax[1].set_ylabel('准确率');ax[1].set_title('正确更新标签也不保证更低NLL');fig.tight_layout(rect=[0,0,1,.85]);save(fig,'09_ablation.png')
    fig,ax=plt.subplots(2,2,figsize=(10,5.5))
    for a,arm,label in zip(ax.ravel(),L['protocol']['arms'],ARM_LABELS):
        for t in tr:
            if t['arm']==arm:a.plot([s['step'] for s in t['states'][:-1]],[s['augmented_train_nll'] for s in t['states'][:-1]],label=str(t['seed']),lw=.85)
        a.set_title(label);a.set(xlabel='更新前步号',ylabel='当步增强批次NLL');a.legend(fontsize=8)
    fig.tight_layout();save(fig,'10_all_traces.png')
    fig,ax=plt.subplots(1,2,figsize=(10,3.4));base=np.array(d['train']['x']);mu=L['normalization']['mean'];sd=L['normalization']['std'];ax[0].hist(base.ravel(),bins=35,color=COLORS[0]);ax[0].axvline(mu,color=COLORS[2],ls='--',label=f'train mean={mu:.3f}');ax[0].set(xlabel='未归一化传感器强度',ylabel='训练像素数',title='尺度由训练数据拟合一次')
    for i,(view,title) in enumerate([('x','原相关'),('swapped','反相关'),('neutral','中性')]):
        means=((np.array(d['test'][view])-mu)/sd).mean((1,2,3));ax[1].scatter(np.arange(128),means,s=8,label=title,alpha=.65)
    ax[1].set(xlabel='同一批测试图编号',ylabel='归一化后每图均值',title='测试不重新居中；保留真实分布差异');h0,l0=ax[0].get_legend_handles_labels();h1,l1=ax[1].get_legend_handles_labels();fig.legend(h0+h1,l0+l1,loc='upper center',bbox_to_anchor=(.5,1.02),ncol=4,frameon=False,fontsize=9);fig.tight_layout(rect=[0,0,1,.84]);save(fig,'11_normalization.png')
    checker=(np.indices((9,9)).sum(0)%2).astype(float);t=torch.tensor(checker)[None,None];no=torch.nn.functional.interpolate(t,size=(3,3),mode='bilinear',align_corners=False,antialias=False)[0,0].numpy();yes=torch.nn.functional.interpolate(t,size=(3,3),mode='bilinear',align_corners=False,antialias=True)[0,0].numpy();fig,ax=plt.subplots(1,3,figsize=(9,3));tile(ax[0],checker,'9×9高频输入');tile(ax[1],no,'3×3：无抗混叠',annot=True);tile(ax[2],yes,'3×3：有抗混叠',annot=True);fig.tight_layout();save(fig,'12_antialias.png')
    print(json.dumps({'figures':12,'font':FONT},ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'figures');main(p.parse_args().output)
