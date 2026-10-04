"""Original mechanism and data-check figures, with explicit Chinese font loading."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle,FancyArrowPatch
import numpy as np
import experiment as e
OUT=Path(__file__).resolve().parent/'figures';OUT.mkdir(exist_ok=True)
fonts=[p for p in font_manager.findSystemFonts() if 'NotoSansCJK-Regular' in p]
if not fonts:raise RuntimeError('Need Noto Sans CJK font for Chinese figure labels')
font_manager.fontManager.addfont(fonts[0]);font=font_manager.FontProperties(fname=fonts[0]).get_name()
plt.rcParams.update({'font.family':font,'font.size':11,'axes.unicode_minus':False})
B,G,O,R='#2865a2','#25836e','#cf851b','#bd4f58'
def canvas(h=4):
    fig,ax=plt.subplots(figsize=(10,h));ax.set(xlim=(0,10),ylim=(0,h));ax.axis('off');return ax

def box(ax,x,y,w,h,text,color=B):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=color+'12',edgecolor=color,lw=1.5))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11,color=color)

def arrow(ax,a,b,color='#728296'):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,lw=1.4,color=color))

def save(name):plt.savefig(OUT/name,dpi=180,bbox_inches='tight',facecolor='white');plt.close()

ax=canvas(4.1)
for x,txt in [(.2,'原始记录\n13 行'),(2.7,'去重与 ID 连接\n12 条观测'),(5.2,'固定实体名单\n6 / 2 / 4'),(7.7,'允许输入张量\n每行两个旋钮')]:box(ax,x,2.4,2.1,.9,txt)
for x in [2.3,4.8,7.3]:arrow(ax,(x,2.85),(x+.35,2.85))
box(ax,2.7,.65,2.1,.85,'09:00 可用\nknob1 / knob2',G);box(ax,5.2,.65,2.1,.85,'09:05 才可用\npost_reading / 标签',R)
arrow(ax,(3.75,1.55),(8.7,2.35),G)
ax.text(8.35,1.05,'晚到字段不得进入 X',ha='center',color=R)
ax.text(5,3.75,'先问每一列何时可用，再问能否变成数值',ha='center',fontsize=14)
save('01_pipeline.png')

ax=canvas(3.4)
for x,txt,c in [(.4,'A01  设备 A  第 1 天\nA02  设备 A  第 2 天\n两次真实观测都保留',G),(3.65,'A02  原始导入\nA02  完全相同再导入\n删除重复导入的副本',B),(6.9,'A02  knob1 = 2\nA02  knob1 = 99\n同 ID 冲突，停止核查',R)]:box(ax,x,1,2.8,1.65,txt,c)
ax.text(5,.3,'设备相同、特征相同、观测相同，是三种不同判断。',ha='center')
save('02_duplicates.png')

fig,axes=plt.subplots(1,2,figsize=(10,4.1))
from matplotlib.colors import ListedColormap
colors=ListedColormap([B,O,G])
entity=np.array([[0]*3]*3+[[1]*3]+[[2]*3]*2)
time=np.array([[0,1,2]]*6)
for ax,grid,title in zip(axes,[entity,time],['目标 A：新设备','目标 B：既有设备的未来']):
    ax.imshow(grid,cmap=colors,vmin=0,vmax=2,aspect='auto');ax.set_xticks([0,1,2],['第 1 天','第 2 天','第 3 天']);ax.set_yticks(range(6),list('ABCDEF'));ax.set_ylabel('设备');ax.set_title(title,pad=12)
    for i in range(6):
        for j in range(3):ax.text(j,i,['训练','验证','测试'][grid[i,j]],ha='center',va='center',color='white',fontsize=10)
fig.subplots_adjust(bottom=.23,wspace=.3)
fig.text(.5,.07,'这是三天的拆分示意；随附主数据只有前两天，正式实验采用实体拆分。',ha='center',fontsize=10)
save('03_splits.png')

clean,_=e.clean_records(e.read_csv(e.ROOT/'data/records_raw.csv'))
train=clean[:6]; X=e.feature_matrix(train)
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
axes[0].hist(X[:,0],bins=[1,3,5,7],color=B,edgecolor='white',rwidth=.94)
axes[0].set(xticks=[1,3,5,7],yticks=[0,1,2,3],xlabel='旋钮 1（格）',ylabel='训练观测条数',title='明确区间的计数图')
for i,r in enumerate(train):
    if r['knob2']!='':
        axes[1].scatter(float(r['knob1']),float(r['knob2']),s=50,color=G);axes[1].annotate(r['record_id'],(float(r['knob1']),float(r['knob2'])),xytext=(4,-13),textcoords='offset points',fontsize=9)
axes[1].set(xlim=(.5,6.6),ylim=(0,68),xlabel='旋钮 1（格）',ylabel='旋钮 2（格）',title='逐条样本核查，A02 缺旋钮 2')
fig.tight_layout();save('04_train_visuals.png')

ax=canvas(4)
box(ax,.3,2.2,4,.9,'仅训练的已观测值\n10 + 30 + 40 + 50 + 60 = 190',B)
box(ax,5.7,2.2,4,.9,'全部已观测值\n再加 70 + 90 + 100 + 110 + 120',R)
arrow(ax,(2.3,2.1),(2.3,1.55),B);arrow(ax,(7.7,2.1),(7.7,1.55),R)
ax.text(2.3,.95,'190 ÷ 5 = 38\n填补 A02 与 D02',ha='center',color=B,fontsize=13)
ax.text(7.7,.95,'680 ÷ 10 = 68\n评估输入反过来改变训练',ha='center',color=R,fontsize=13)
ax.text(5,3.65,'不看标签也可能越过信息边界',ha='center',fontsize=14)
save('05_mean_leakage.png')

fig,ax=plt.subplots(figsize=(10,3.4))
ax.axvspan(0,1,color=B,alpha=.10,label='训练极值之间')
ax.axvline(0,color=B,ls=':');ax.axvline(1,color=B,ls=':')
values=[0,.2,.4,.6,.8,1,1.2,1.4]
ax.scatter(values,[1]*6+[.5]*2,c=[B]*6+[O]*2,s=70)
for x,t in zip(values,list('12345678')):ax.annotate(t+' 格',(x,1 if int(t)<=6 else .5),xytext=(0,10),textcoords='offset points',ha='center')
ax.set(xlim=(-.12,1.55),ylim=(.15,1.5),yticks=[.5,1],yticklabels=['验证 D','训练 A/B/C'],xlabel='缩放后 z = (x - 1) / 5',title='沿用训练刻度，验证值超过 1 不等于代码错误')
ax.legend(loc='upper left');fig.tight_layout();save('06_scale.png')

ax=canvas(4)
for y,rid,x,yv in [(2.7,'A01',1,2),(1.85,'A02',2,5),(1,'B01',3,12)]:
    box(ax,.4,y-.3,2,.6,f'{rid}  x1={x}',B)
for y,rid,yv in [(2.7,'B01',12),(1.85,'A02',5),(1,'A01',2)]:box(ax,7.6,y-.3,2,.6,f'{rid}  y={yv}',G)
for y1,y2 in [(2.7,1),(1.85,1.85),(1,2.7)]:arrow(ax,(2.5,y1),(7.5,y2),G)
ax.text(5,3.6,'按 record_id 连接，文件行号不必相同',ha='center',fontsize=14)
ax.text(5,.22,'若按当前位置连接，A01 会错误得到 12；shape 仍然完全合法。',ha='center',color=R)
save('07_alignment.png')
