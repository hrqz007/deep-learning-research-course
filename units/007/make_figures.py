"""Rebuild eight original scalar-algebra figures; plotting only, no NumPy API."""
from pathlib import Path
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch
OUT=Path(__file__).resolve().parent/'figures'
OUT.mkdir(exist_ok=True)
fonts=[p for p in font_manager.findSystemFonts() if 'NotoSansCJK-Regular' in p]
if fonts:
    font_manager.fontManager.addfont(fonts[0])
    font=font_manager.FontProperties(fname=fonts[0]).get_name()
else:
    raise RuntimeError('Install a CJK font such as Noto Sans CJK before rebuilding figures')
plt.rcParams.update({'font.family':[font,'DejaVu Sans'],'font.size':13,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
B,G,O,R='#2563a6','#25806c','#d57b18','#bc4951'
def save(name):
    plt.savefig(OUT/name,dpi=180,bbox_inches='tight',facecolor='white');plt.close()
def grid(lo,hi,n=161):return [lo+(hi-lo)*i/(n-1) for i in range(n)]
def axes(ax,xlabel='输入 x（无量纲）',ylabel='输出 y（无量纲）'):
    ax.set(xlabel=xlabel,ylabel=ylabel);ax.axhline(0,color='#9aa5b1',lw=.8);ax.axvline(0,color='#9aa5b1',lw=.8);ax.grid(alpha=.15)
def box(ax,x,y,w,h,text,color=B):
    ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor=color,lw=1.8));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=13)
def arrow(ax,start,end):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=16,color='#687788',lw=1.5))
fig,ax=plt.subplots(figsize=(10,3));ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off')
for x,text in [(.15,'2x + 1 = 7'),(3.6,'2x = 6'),(7.05,'x = 3')]:box(ax,x,1.5,2.8,.9,text)
for start,end in [(2.98,3.55),(6.45,7)]:arrow(ax,(start,1.95),(end,1.95))
ax.text(3.25,2.65,'两边减1',ha='center');ax.text(6.7,2.65,'两边除以2',ha='center');ax.text(5,.85,'代回原式：2 × 3 + 1 = 7',ha='center',color=G)
ax.text(5,.22,'一般 ax+b=c：a≠0 才能除以a；a=0 必须另行判断',ha='center',color=R);save('01_equation.png')
fig,ax=plt.subplots(figsize=(9.5,4));xx=grid(-1,7)
ax.plot(xx,[2*x+1 for x in xx],ls='--',color='#9aa5b1',label='表达式的数学延伸')
ax.plot([0,5],[1,11],color=B,lw=3,label='任务允许：0 ≤ x ≤ 5');ax.scatter([0,5],[1,11],color=B,s=70,zorder=4)
ax.axvspan(0,5,color=B,alpha=.07);ax.annotate('(0, 1)',(0,1),xytext=(.3,3));ax.annotate('(5, 11)',(5,11),xytext=(5.15,10))
ax.set(xlim=(-1,7),ylim=(-2,16));axes(ax,'旋钮位置 x（格）','输出 f(x)（读数单位）');ax.legend(loc='upper left',fontsize=11);save('02_domain.png')
fig,aa=plt.subplots(1,2,figsize=(10,3.8));cm=[1,2,3];mm=[10,20,30]
aa[0].plot(cm,[3,5,7],'o-',color=B,lw=2,label='2x + 1');aa[0].set(ylim=(0,10),xticks=cm);axes(aa[0],'长度 x（厘米）','输出（读数单位）');aa[0].legend()
aa[1].plot(mm,[3,5,7],'o-',color=G,lw=2,label='0.2u + 1：等价');aa[1].plot(mm,[21,41,61],'x--',color=R,label='2u + 1：错误沿用');aa[1].set(ylim=(0,68),xticks=mm);axes(aa[1],'同一长度 u（毫米）','输出（读数单位）');aa[1].legend(fontsize=11,loc='upper left');fig.tight_layout();save('03_units.png')
fig,ax=plt.subplots(figsize=(9,4));xx=grid(0,4)
ax.plot(xx,[2*x+1 for x in xx],color=B,lw=2,label='2x + 1：每步加2');ax.plot(xx,[2**x for x in xx],color=O,lw=2,label='2的x次幂：每步乘2')
ax.scatter(range(5),[1,3,5,7,9],color=B);ax.scatter(range(5),[1,2,4,8,16],color=O)
ax.set(xticks=range(5),ylim=(0,18));axes(ax);ax.legend(loc='upper left');save('04_growth.png')
fig,ax=plt.subplots(figsize=(9,4.2));xx=grid(.125,8,400);ax.axvspan(-1,0,color=R,alpha=.12,label='x ≤ 0：无实数对数')
ax.plot(xx,[math.log2(x) for x in xx],color=G,lw=2,label='y = log₂ x，x > 0');px=[.25,.5,1,2,4,8];py=[-2,-1,0,1,2,3];ax.scatter(px,py,color=G,s=38,zorder=4)
for x,y,label,xy in [(1,0,'(1, 0)',(1.4,-.75)),(2,1,'(2, 1)',(2.2,.45)),(4,2,'(4, 2)',(4.2,1.25)),(8,3,'(8, 3)',(6.7,3.35))]:ax.annotate(label,(x,y),xytext=xy,fontsize=11)
ax.text(2.8,-2.8,'例如：2³ = 8 反读为 log₂8 = 3',fontsize=12);ax.set(xlim=(-1,8.8),ylim=(-3.5,4.4));axes(ax);ax.legend(loc='upper left',fontsize=11);save('05_log.png')
fig,aa=plt.subplots(1,3,figsize=(10,3.6));values=[(5,4),(math.log2(5),2),(math.log2(3),2)];titles=['先求原均值','原均值后取 log₂','每项 log₂ 后平均']
for ax,(a,b),title,win in zip(aa,values,titles,['B','B','A']):
    ax.bar(['A','B'],[a,b],color=[B,O],width=.55)
    for i,y in enumerate([a,b]):ax.text(i,y+.08,f'{y:.3f}'.rstrip('0').rstrip('.'),ha='center',fontsize=12)
    ax.set(title=title,ylim=(0,max(a,b)*1.32),ylabel='对应目标值（越小越好）');ax.text(.5,.93,'选 '+win,transform=ax.transAxes,ha='center',color=G);ax.grid(axis='y',alpha=.15)
fig.tight_layout();save('06_objectives.png')
fig,ax=plt.subplots(figsize=(10,3.5));ax.set(xlim=(0,10),ylim=(0,3.5));ax.axis('off')
for y,label,items in [(2.15,'f(g(2))',[(.15,'输入 2'),(2.5,'g 平方'),(4.85,'得到 4'),(7.2,'f 乘2加1\n得到 9')]),(.45,'g(f(2))',[(.15,'输入 2'),(2.5,'f 乘2加1'),(4.85,'得到 5'),(7.2,'g 平方\n得到 25')])]:
    ax.text(.15,y+1.12,label,color=B,fontsize=13)
    for x,t in items:box(ax,x,y,2.1,.9,t,G if y<1 else B)
    for a,b in [(2.3,2.45),(4.65,4.8),(7,7.15)]:arrow(ax,(a,y+.45),(b,y+.45))
save('07_composition.png')
fig,aa=plt.subplots(2,2,figsize=(10,5.5));spec=[('仿射 2x+1',lambda x:2*x+1,B,(-2,3)),('指数 2的x次幂',lambda x:2**x,O,(-2,3)),('对数 log₂x，仅 x>0',math.log2,G,(.125,4)),('分段 max(0,x)',lambda x:max(0,x),R,(-2,3))]
for ax,(title,fn,color,(lo,hi)) in zip(aa.flat,spec):
    xx=grid(lo,hi);ax.plot(xx,[fn(x) for x in xx],color=color,lw=2);axes(ax,'输入 x','输出');ax.set_title(title,fontsize=13);ax.tick_params(labelsize=11)
aa[1,0].axvspan(-.4,0,color=R,alpha=.12);aa[1,0].set_xlim(-.4,4.2)
aa[1,1].scatter([0],[0],color=R,s=50,zorder=4);aa[1,1].text(-1.8,1.2,'边界 (0,0)',fontsize=11);aa[1,1].set_ylim(-.35,3.4)
fig.tight_layout();save('08_function_family.png')
print('Saved 8 original figures')
