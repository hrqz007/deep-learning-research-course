"""All original diagrams and charts are deterministic; no downloaded art."""
from pathlib import Path
import argparse,json,tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
from reference import iou,hand_ledger,smooth_l1,detection_ap
from matplotlib import font_manager
# Linux TTC collections expose only the first (JP) face to Matplotlib by default.
# Extract the installed SC face into an isolated cache; never silently emit tofu.
try:
    font_manager.findfont('Noto Sans CJK SC', fallback_to_default=False)
except ValueError:
    fontfile=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if not fontfile.exists():
        raise RuntimeError('请安装Noto Sans CJK SC字体，然后重新生成图；不接受缺字输出')
    from fontTools.ttLib import TTCollection
    cache=Path(tempfile.gettempdir())/'dl046-font-cache';cache.mkdir(parents=True,exist_ok=True)
    extracted=cache/'NotoSansCJKSC-Regular.ttf'
    if not extracted.exists():
        collection=TTCollection(str(fontfile))
        font=next(f for f in collection.fonts if f['name'].getDebugName(1)=='Noto Sans CJK SC')
        font.save(extracted)
    font_manager.fontManager.addfont(str(extracted))
ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'Noto Sans CJK SC','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'figure.dpi':150,'savefig.dpi':150})
COL=['#126782','#ed8b23','#813d9c'];NAMES={'pixel':'逐像素线性','cnn_bce':'CNN / BCE','cnn_weighted':'CNN / 加权 BCE'}

def main(results,output):
    r=json.loads(Path(results).read_text());a=np.load(Path(results).parent/'arrays.npz');out=Path(output);out.mkdir(parents=True,exist_ok=True);index=[]
    def save(name,desc):
        
        if name=='09_all_traces.png':plt.gcf().subplots_adjust(bottom=.35,top=.90,left=.06,right=.99,wspace=.36)
        else:plt.tight_layout()
        plt.savefig(out/name,bbox_inches='tight',facecolor='white');plt.close();index.append({'file':name,'description':desc,'origin':'original code, specified toy example or retained experiment data'})
    def mask(ax,x,title,cmap='gray'):
        ax.imshow(x,cmap=cmap,vmin=0,vmax=1,interpolation='nearest',extent=(0,16,16,0));ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
    x=np.load(ROOT/'data/test_x.npy')[:,0];y=np.load(ROOT/'data/test_y.npy')[:,0];meta=json.loads((ROOT/'data/metadata.json').read_text())
    fig,axes=plt.subplots(1,4,figsize=(10,2.7));ix=2;mask(axes[0],x[ix],'输入图像');mask(axes[1],x[ix],'整图标签：有目标')
    mask(axes[2],x[ix],'检测：两条框记录')
    for j,(l,t,rr,b) in enumerate(meta['splits']['test']['records'][ix]['boxes']):axes[2].add_patch(Rectangle((l,t),rr-l,b-t,fill=False,lw=2,color=COL[j]))
    mask(axes[3],y[ix],'语义分割：前景并集',cmap='Blues');save('01_tasks.png','相同输入，分类、检测与语义分割的输出对象不同')
    s=np.linspace(-3,3,601);v=np.where(abs(s)<2,(2-abs(s))/(2+abs(s)),0);d=np.where(abs(s)>2,0.,np.where((abs(s)<2)&(abs(s)>0),-4*np.sign(s)/(2+abs(s))**2,np.nan))
    fig,axes=plt.subplots(1,2,figsize=(9,2.8));axes[0].plot(s,v);axes[0].set(xlabel='预测框水平位移 s',ylabel='IoU',title='等大 2×2 框：交集 / 并集');axes[1].plot(s,d);axes[1].set(xlabel='位移 s',ylabel='分支内导数',title='不相交区间 IoU 梯度为 0');axes[1].axhline(0,color='gray',lw=.8);save('02_iou_branches.png','IoU分段几何及不可微边界')
    z=np.linspace(-2.5,2.5,200);l,g=smooth_l1(z);fig,axes=plt.subplots(1,3,figsize=(10,2.8));axes[0].plot(z,l);axes[0].set(title='Smooth L1，β=1',xlabel='编码残差 r',ylabel='损失');axes[1].plot(z,g);axes[1].set(title='小残差线性梯度 / 大残差截幅',xlabel='r',ylabel='局部导数');axes[2].axis('off');axes[2].text(.02,.87,'锚框 [0,0,2,2]\n中心 (1,1)，宽高 (2,2)\n\n真框 [1,0,5,2]\n中心 (3,1)，宽高 (4,2)\n\n目标编码 (1,0,ln 2,0)',va='top',fontsize=11);save('03_box_regression.png','框编码与稳健的分段回归损失')
    fig,axes=plt.subplots(1,2,figsize=(9,3));b=[[0,0,3,3],[.2,.2,3.2,3.2],[2.5,0,5.5,3]]
    for ax in axes:
        ax.set(xlim=(-.5,6),ylim=(4,-1),aspect='equal');ax.set_xticks([]);ax.set_yticks([])
        for j,bb in enumerate(b):l,t,rr,bt=bb;ax.add_patch(Rectangle((l,t),rr-l,bt-t,fill=False,color=COL[j],lw=2));ax.text(l+.05,t+.4,f'P{j+1}',color=COL[j])
    axes[0].set_title('NMS：同类、按分数逐个保留');axes[0].text(0,3.75,'分数 .9 / .8 / .7：保留 P1、P3',fontsize=10);axes[1].set_title('评估匹配：一个真框最多匹配一次');axes[1].text(0,3.75,'P2 即使定位正确，也可能是重复 FP',fontsize=10);save('04_matching_nms.png','NMS不读取真值；评估匹配读取真值且不可重复使用')
    gt=[{'image_id':'a','box':[0,0,2,2]},{'image_id':'b','box':[0,0,2,2]}];pred=[dict(gt[0],score=.9),dict(gt[0],score=.8),dict(gt[1],score=.7)];ap=detection_ap(gt,pred)
    fig,axes=plt.subplots(1,2,figsize=(9,2.8));axes[0].plot(ap['recall'],ap['precision'],'o--',label='原始 P/R 点');axes[0].step([0,.5,1],[1,1,2/3],where='pre',label='单调包络');axes[0].set(xlim=(0,1.05),ylim=(0,1.1),xlabel='Recall',ylabel='Precision',title='TP、重复 FP、TP');axes[0].legend(fontsize=9);axes[1].axis('off');axes[1].text(.04,.87,'全点积分 AP = 5/6 = 0.833333\n\n101 点 AP = (51 + 50×2/3)/101\n                     = 0.834983\n\nAP50 固定定位门槛 0.50\nmAP 还须说明类别与门槛平均',va='top',fontsize=11);save('05_ap_convention.png','相同排序在不同AP采样约定下可有不同结果')
    fig,ax=plt.subplots(figsize=(10,2.8));ax.axis('off');coords=[(.03,.22,.22,.6,'C2：16×16\n细节多，步幅小'),(.37,.34,.19,.44,'C3：8×8\n更大感受野'),(.72,.44,.15,.27,'C4：4×4\n更粗位置网格')]
    for l,t,w,h,s in coords:ax.add_patch(Rectangle((l,t),w,h,fc='#e5f1f7',ec=COL[0],lw=2));ax.text(l+w/2,t+h/2,s,ha='center',va='center')
    ax.annotate('',xy=(.37,.55),xytext=(.25,.55),arrowprops={'arrowstyle':'->'});ax.annotate('',xy=(.72,.55),xytext=(.56,.55),arrowprops={'arrowstyle':'->'});ax.text(.5,.02,'概念图：多尺度融合需要对齐空间位置。主实验没有下采样，也不是 FPN / U-Net。',ha='center');save('06_multiscale.png','多尺度特征图的语义与坐标权衡')
    h=hand_ledger();fig,axes=plt.subplots(1,2,figsize=(10,3));pixels=[p for s in h[0]['samples'] for p in s['pixels']];axes[0].bar(np.arange(4)-.16,[p['dw_contribution'] for p in pixels],.32,label='∂L/∂w 的局部贡献');axes[0].bar(np.arange(4)+.16,[p['db_contribution'] for p in pixels],.32,label='∂L/∂b 的局部贡献');axes[0].set_xticks(range(4),['A0','A1','B0','B1']);axes[0].axhline(0,color='gray',lw=.8);axes[0].legend(fontsize=8);axes[0].set_title('共享参数：四条像素路径求和');axes[1].plot([0,1,2],[v['loss'] for v in h],'o-');axes[1].set(xlabel='同步更新次数',ylabel='四像素平均 BCE',title='全部参数一起更新，再重新前向');save('07_hand_gradient.png','两个图像四个像素的梯度贡献与重新计算的损失')
    fig,axes=plt.subplots(2,4,figsize=(9,4.6))
    for j in range(4):mask(axes[0,j],x[j],['空图','一个目标','重叠目标','分离目标'][j]);mask(axes[1,j],y[j],'真值前景并集','Blues')
    save('08_data.png','四种标注情况均存在于每个数据拆分')
    fig,axes=plt.subplots(1,3,figsize=(11,4.2))
    for j,arm in enumerate(['pixel','cnn_bce','cnn_weighted']):
        for c in r['candidates']:
            if c['arm']==arm:
                style={.1:':',1.:'-',10.:'--'}[c['lr']];axes[j].plot(np.arange(len(c['history']))+1,[h['validation_iou_after_update'] for h in c['history']],style,lw=1,label=f"lr={c['lr']:g}, s={c['seed']}")
        axes[j].set(xlabel='已执行更新数',ylabel='验证前景 micro IoU',ylim=(-.02,.85),title=NAMES[arm]);axes[j].legend(fontsize=8,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.27))
    save('09_all_traces.png','21次训练完整验证曲线，虚线压力实验也保留')
    fig,axes=plt.subplots(1,2,figsize=(10,3.2));arms=['pixel','cnn_bce','cnn_weighted'];labels=['逐像素','CNN BCE','CNN 加权','3×3均值','强度阈值']
    for j,split in enumerate(['test','shift']):
        for k,arm in enumerate(arms):axes[j].scatter([k]*3,[s['metrics'][split]['foreground_iou_micro'] for s in r['selected'] if s['arm']==arm],s=40,alpha=.7,color=COL[k])
        axes[j].scatter([3,4],[r['baselines'][b]['metrics'][split]['foreground_iou_micro'] for b in ['mean3x3','intensity']],marker='s',color='#536b42');axes[j].set_xticks(range(5),labels,rotation=15);axes[j].set(ylabel='前景 micro IoU',ylim=(-.02,.85),title=['封存同分布测试','独立亮背景偏移测试'][j]);axes[j].grid(axis='y',alpha=.2)
    save('10_results.png','每个点是预定种子；确定性逐像素模型的三个点重合')
    key='cnn_bce_lr1_s11';logits=a[key+'_test_logits'][:,0];pred=logits>=0;prob=1/(1+np.exp(-logits));fig,axes=plt.subplots(4,4,figsize=(8,7.3))
    for j,ix in enumerate([0,1,2,3]):
        for row,img in enumerate([x[ix],y[ix],prob[ix],pred[ix].astype(float)]):mask(axes[row,j],img,['输入','真值','前景概率','阈值预测'][row]+f' / {ix:02}',cmap='gray' if row==0 else 'Blues')
    save('11_predictions.png','固定展示test-000至003，不挑选最好样本；模型种子11')
    # Fixed ordering: all 48 images, no worst-case omission or cherry-picking.
    fig,ax=plt.subplots(figsize=(10,3.8));matrix=np.array([s['metrics']['test']['per_image_iou'] for s in r['selected']]);im=ax.imshow(matrix,vmin=0,vmax=1,aspect='auto',cmap='viridis');ax.set_yticks(range(9),[s['id'].replace('_lr',' / η=').replace('_s',' / s=') for s in r['selected']],fontsize=8);ax.set(xlabel='全部测试图像索引（每4张依次空图/单目标/重叠/分离）',title='逐图 IoU：空真值且空预测按 1 计');fig.colorbar(im,ax=ax,label='逐图 IoU');save('12_all_test_images.png','九个冻结选择模型在48张测试图上的全部逐图IoU')
    fig,axes=plt.subplots(1,2,figsize=(10,3));
    for b,c in zip(['intensity','mean3x3'],COL):axes[0].plot([v['threshold'] for v in r['baselines'][b]['search']],[v['validation_iou'] for v in r['baselines'][b]['search']],'o-',ms=3,label=b,color=c)
    axes[0].set(xlabel='验证集候选阈值',ylabel='前景 micro IoU',title='全部 50 个阈值候选');axes[0].legend();vals=[]
    for arm in arms:vals.append(np.mean([s['metrics']['test']['empty_false_positive_rate'] for s in r['selected'] if s['arm']==arm]))
    axes[1].bar(range(3),vals,color=COL);axes[1].set_xticks(range(3),['逐像素','CNN BCE','CNN 加权']);axes[1].set(ylim=(0,1),ylabel='空图中至少1个假阳性的比例',title='单报像素准确率会漏掉的失败');save('13_threshold_empty.png','阈值搜索和空图误报揭示损失与决策的区别')
    (out/'figure_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n');print(f'{len(index)} original figures')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',default=str(ROOT/'outputs/results.json'));p.add_argument('--output',default=str(ROOT/'figures'));a=p.parse_args();main(a.results,a.output)
