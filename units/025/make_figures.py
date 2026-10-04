"""Original scientific figures. Validate default inputs AND current results first."""
from pathlib import Path
import io, os
import numpy as np
import experiment as e

def main():
    inputs,result,_=e.verify_teaching_artifacts()
    os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager, FontProperties
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if Path(font).exists():
        fontManager.addfont(font); plt.rcParams['font.family']=FontProperties(fname=font).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    blue='#2675b7'; orange='#e07c35'; green='#278b77'; purple='#925bb1'; gray='#64748b'
    payload={}
    def save(fig,name):
        fig.tight_layout(pad=1.3); b=io.BytesIO(); fig.savefig(b,format='png',dpi=170); payload[name]=b.getvalue(); plt.close(fig)
    def fmt(ax,x='x',y='y'):
        ax.set_xlabel(x);ax.set_ylabel(y);ax.grid(alpha=.2)
    X,y=inputs[:2]
    fig,ax=plt.subplots(figsize=(9.6,3.2));ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,3)
    boxes=[(.2,'输入 X','n × 2\n每行一条样本',blue),(2.7,'仿射 Z(1)','X @ W(1) + b(1)\nn × 2',orange),(5.2,'隐藏 H(1)','逐元素 max(0, Z(1))\nn × 2',green),(7.7,'输出 Z(2)','H(1) @ W(2) + b(2)\nn × 1',purple)]
    for x,title,body,c in boxes:
        ax.add_patch(plt.Rectangle((x,.7),2.1,1.7,facecolor=c+'16',edgecolor=c,lw=1.8));ax.text(x+1.05,2.06,title,ha='center',weight='bold',color=c);ax.text(x+1.05,1.38,body,ha='center',va='center',fontsize=10)
        if x<7:ax.annotate('',xy=(x+2.45,1.55),xytext=(x+2.13,1.55),arrowprops={'arrowstyle':'->','color':gray,'lw':1.7})
    ax.text(5,.18,'2 个带参数层；1 个隐藏层；权重 4 + 2，偏置 2 + 1，共 9 个标量',ha='center',color=gray)
    save(fig,'01_forward_shapes.png')
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.6));t=np.linspace(-1,2,200)
    axs[0].plot(t,2*t+1,label='u = 2x + 1',color=blue);axs[0].plot(t,-3*(2*t+1)+4,label='v = -3u + 4',color=orange);axs[0].plot(t,-6*t+1,'--',label='合并 v = -6x + 1',color=green);fmt(axs[0],'x','输出');axs[0].legend(fontsize=9);axs[0].set_title('两次仿射仍是一条直线')
    axs[1].plot(t,-3*np.maximum(2*t+1,0)+4,color=purple,lw=2,label='v = -3 ReLU(2x+1) + 4');axs[1].axvline(-.5,color=gray,ls=':');fmt(axs[1],'x','输出');axs[1].set_title('插入 ReLU 后有了折点');axs[1].legend(fontsize=9)
    save(fig,'02_affine_composition.png')
    H=e.forward(X,e.xor_layers())[1][0]['h'];fig,axs=plt.subplots(1,2,figsize=(9.6,3.6))
    for i,(a,b) in enumerate(X):axs[0].scatter(a,b,color=orange if y[i] else blue,s=140,edgecolor='white',zorder=3);axs[0].annotate(inputs[3][i]+f': y={int(y[i])}',(a,b),xytext=(7,6),textcoords='offset points',fontsize=9)
    axs[0].set(xlim=(-.25,1.45),ylim=(-.25,1.35),title='输入空间：两条对角线交叉');fmt(axs[0],'x1','x2')
    for i,(a,b) in enumerate(H):axs[1].scatter(a,b,color=orange if y[i] else blue,s=150,edgecolor='white',zorder=3)
    for txt,pt,off in [('a: y=0',(0,0),(8,8)),('b,c: y=1',(1,0),(8,8)),('d: y=0',(2,1),(-65,10))]:axs[1].annotate(txt,pt,xytext=off,textcoords='offset points',fontsize=9)
    u=np.linspace(0,2.3,100);axs[1].plot(u,.5*u-.25,color=purple,label='2h1 - 4h2 - 1 = 0');axs[1].set(xlim=(-.25,2.4),ylim=(-.45,1.35),title='隐藏空间：用一条直线分开');fmt(axs[1],'h1 = ReLU(x1+x2)','h2 = ReLU(x1+x2-1)');axs[1].legend(fontsize=9,loc='upper left')
    save(fig,'03_hidden_representation.png')
    g=np.linspace(-.2,1.2,241);xx,yy=np.meshgrid(g,g);grid=np.column_stack([xx.ravel(),yy.ravel()]);
    zn=e.forward(grid,e.xor_layers())[0][:,0].reshape(xx.shape);zl=e.forward(grid,e.xor_layers(),activation='identity')[0][:,0].reshape(xx.shape)
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.8))
    for ax,Z,title in zip(axs,[zl,zn],['去掉 ReLU：固定同一组权重，3/4 正确','保留 ReLU：固定同一组权重，4/4 正确']):
        ax.contourf(xx,yy,Z>=0,levels=[-.5,.5,1.5],colors=[blue+'35',orange+'35']);ax.contour(xx,yy,Z,levels=[0],colors=purple,linewidths=2)
        ax.scatter(X[:,0],X[:,1],c=[orange if v else blue for v in y],s=100,edgecolor='black');fmt(ax,'x1','x2');ax.set_title(title,fontsize=10);ax.set_aspect('equal')
    save(fig,'04_decision_boundaries.png')
    s=np.linspace(0,2,401);fig,axs=plt.subplots(1,2,figsize=(9.6,3.5))
    axs[0].plot(s,s,label='h1 = s',color=blue);axs[0].plot(s,np.maximum(s-1,0),label='h2 = max(0,s-1)',color=green);axs[0].axvline(1,color=gray,ls=':');fmt(axs[0],'s = x1 + x2','隐藏坐标');axs[0].legend();axs[0].set_title('每个单元贡献一段斜坡')
    axs[1].plot(s,2*s-4*np.maximum(s-1,0)-1,color=purple,lw=2);axs[1].axhline(0,color=gray,ls=':');axs[1].scatter([0,1,2],[-1,1,-1],c=[blue,orange,blue],s=85,zorder=3);axs[1].axvspan(.5,1.5,color=orange,alpha=.1);fmt(axs[1],'s = x1 + x2','logit z');axs[1].set_title('加权组合产生两个分类边界')
    save(fig,'05_piecewise_logits.png')
    fig,axs=plt.subplots(1,3,figsize=(9.6,3.3));t=np.linspace(0,1,513)
    for ax,d,c in zip(axs,[1,2,3],[blue,green,purple]):ax.plot(t,e.triangle(t,d),color=c,lw=2);fmt(ax,'t','输出');ax.set_title(f'T 复合 {d} 次：{2**(d-1)} 个峰');ax.set_ylim(-.08,1.15)
    save(fig,'06_depth_folding.png')
    searches=result[2];fig,axs=plt.subplots(1,2,figsize=(9.6,3.5))
    losses=sorted(r['loss'] for r in searches[:125]);axs[0].plot(range(1,126),losses,color=blue);axs[0].axhline(np.log(2),color=gray,ls='--',label='log(2) = 0.693147');axs[0].set_title('125 个仿射候选，按损失排序');fmt(axs[0],'候选排序（不是迭代）','平均交叉熵');axs[0].legend(fontsize=9)
    scales=[r['scale'] for r in searches[125:]];vals=[r['loss'] for r in searches[125:]];axs[1].plot(scales,vals,'o-',color=orange);axs[1].set_yscale('log');axs[1].set_title('仅搜索一个输出尺度 γ');fmt(axs[1],'γ（5 个事先规定的候选）','平均交叉熵（对数刻度）')
    save(fig,'07_bounded_search.png')
    g=np.linspace(0,1,201);xx,yy=np.meshgrid(g,g);z1=2*(xx+yy)-4*np.maximum(xx+yy-1,0)-1;z2=2*np.abs(xx-yy)-1
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.8))
    for ax,Z,title in zip(axs,[z1,z2],['规则 A：两个输入求和再折叠','规则 B：两个输入做差取绝对值']):
        ax.contourf(xx,yy,Z>=0,levels=[-.5,.5,1.5],colors=[blue+'35',orange+'35']);ax.contour(xx,yy,Z,levels=[0],colors=purple);ax.scatter(X[:,0],X[:,1],c=[orange if v else blue for v in y],s=85,edgecolor='black');ax.scatter([.5],[.5],marker='*',s=150,color=green,edgecolor='black');ax.set_aspect('equal');ax.set_xlim(-.08,1.08);ax.set_ylim(-.08,1.08);fmt(ax,'x1','x2');ax.set_title(title,fontsize=10)
    save(fig,'08_unseen_ambiguity.png')
    out=e.safe_destination(e.ROOT/'figures',payload)
    out.mkdir(exist_ok=True)
    for name,data in payload.items():(out/name).write_bytes(data)
    print('Verified inputs/results; rendered and serialized all',len(payload),'figures before writing.')
if __name__=='__main__':main()
