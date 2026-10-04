"""Rebuild original diagrams only after checking the fixed report inputs."""
from pathlib import Path
import hashlib,json,csv,os
ROOT=Path(__file__).resolve().parent
EXPECTED={'data/config.json': '415dfe8ed8750b983b454ab38629ee36fb282f3e9449c955065445abc9eca238', 'data/samples.csv': 'e26f750b0e4c25f0dcdd53100a395a86b21f96d056b91365010da4cf68eb6c4c', 'data/split.json': '43298159b4808eb3b80fe841d347f26543cbb95212f657fa7a5ea86a1fd2d98f', 'outputs/checkpoint-epoch-04.pt': 'e548b062fdfe9e7bbb943e3069e1a31932d8afe9d60342ab6bf8e74e8c88e414', 'outputs/final-state.json': '493efb4a4cf67dc579a274123ef4154cd51f7dc7f1ea1f7f724d2156d1abb684', 'outputs/first-step-trace.json': '5d7edef4392ea939d53d66d76de245390a56934fe9c3c4153fa7ea92e9df4632', 'outputs/history.csv': '9a9c11c9068c0d3caa697aa154d7a7d1a62c285064b5b4a789089c29de72b575', 'outputs/orders.json': '2f843bdeabc35b1d6075d24e6a0eae82cf6367014b98fd98a3728f81a31e6ea4', 'outputs/semantics.json': '649b502af008fbff9891fa448b2a4fef2971f17122a086e9fc9e3c598d5b6abd', 'outputs/summary.json': 'a2c6a776f53c4d10018efbee8230e8e812d49d7942ce6aa8b6a4dc3d701e26dc', 'outputs/test-predictions.json': 'e91924ef3f77c74190e04d9bfec09f88a19424ab39ea578baeec320290e5cd1b'}
def check_artifacts():
 for name,digest in EXPECTED.items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
   raise ValueError('Fixed report changed: '+name)
check_artifacts()
# Do not import rendering dependencies or create files before input validation.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
from matplotlib.patches import FancyBboxPatch
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if Path(FONT).exists():
 fontManager.addfont(FONT);plt.rcParams['font.family']=FontProperties(fname=FONT).get_name()
plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
C=['#176a9a','#c66c24','#29906c','#995499','#cf4747']
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
s=json.loads((ROOT/'outputs/summary.json').read_text())
h=list(csv.DictReader((ROOT/'outputs/history.csv').open()))
def save(fig,name):
 fig.savefig(OUT/name,bbox_inches='tight',facecolor='white');plt.close(fig)
def setup(title,size=(9,3.7)):
 f,a=plt.subplots(figsize=size);a.set_title(title,loc='left',fontsize=14,pad=14);a.set_xlim(0,10);a.set_ylim(0,5);a.axis('off');return f,a
def box(a,x,y,w,h,text,color):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',fc=color,ec='none',alpha=.12))
 a.text(x+w/2,y+h/2,text,ha='center',va='center',color='#18202a',fontsize=11)
def arrow(a,x,y,xx,yy,color='#64748b'):
 a.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':color,'lw':1.7})
f,a=setup('训练、验证、测试分别改变什么')
for x,txt,col in [(0.1,'训练 45 行\n更新参数与动量\n拟合标准化统计量',C[0]),(3.5,'验证 15 行\n比较候选 epoch\n记录最佳权重',C[1]),(6.9,'测试 12 行\n选择完成后评价一次\n不得反馈调参',C[2])]:box(a,x,1.8,2.8,2.0,txt,col)
arrow(a,2.95,2.8,3.45,2.8);arrow(a,6.35,2.8,6.85,2.8)
a.text(5,.7,'拆分 ID 固定；只有训练行参与 mean 与 scale 的计算',ha='center')
save(f,'01_responsibilities.png')
f,a=setup('注册决定管理范围，求导决定计算图依赖')
box(a,.2,2.3,2.5,1.6,'SmallRegressor\nnn.Module',C[0])
for y,text,col in [(3.5,'parameters：W1(4,2), b1(4), W2(1,4), b2(1)',C[2]),(2,'buffers：mean(2), scale(2)',C[1]),(.5,'Tanh / Dropout：无可学习参数，但有模式',C[3])]:
 box(a,3.5,y,6.2,1,text,col);arrow(a,2.75,3,3.4,y+.5)
a.text(1.45,.8,'17 个可学习标量\n4 个固定统计标量',ha='center')
save(f,'02_registration.png')
f,a=setup('同一组样本经过两层索引组织成批次')
box(a,.1,2.8,2.6,1.4,'原始 72 行\nid → x(2), y(1)',C[0]);box(a,3.5,2.8,2.6,1.4,'Rows 训练子集\n局部索引 → 原始 ID',C[2]);box(a,7,2.8,2.8,1.4,'DataLoader\n每轮打乱局部索引',C[1])
arrow(a,2.8,3.5,3.4,3.5);arrow(a,6.2,3.5,6.9,3.5)
box(a,1.4,.6,7.3,1.3,'45 = 7 + 7 + 7 + 7 + 7 + 7 + 3\n每批 X(B,2)，y(B,1)，id(B)；保留最后 3 行',C[3]);arrow(a,8.4,2.7,8.4,2)
save(f,'03_batches.png')
f,a=setup('一次训练 step 与一次验证的时序',(9.5,4))
train=['train()','取批次','zero_grad','forward\nloss','backward','step']
for i,txt in enumerate(train):
 x=.1+i*1.64;box(a,x,3,1.45,1.1,txt,C[0])
 if i<5:arrow(a,x+1.49,3.55,x+1.6,3.55)
for x,w,txt in [(.1,2,'记录旧模式'),(2.5,2.3,'eval() + no_grad'),(5.2,2.4,'只前向与累计误差'),(8,1.8,'恢复模式')]:box(a,x,.7,w,1.1,txt,C[2])
for x,xx in [(2.15,2.45),(4.85,5.15),(7.65,7.95)]:arrow(a,x,1.25,xx,1.25)
a.text(5,2.3,'验证不调用 backward / optimizer.step；检查参数、buffer 和已有梯度未改变',ha='center',fontsize=10)
save(f,'04_timeline.png')
f,axs=plt.subplots(1,2,figsize=(9,3.5))
axs[0].bar(['2 行批次','3 行批次'],[1,4],color=C[:2]);axs[0].set_ylabel('各批次平均平方误差');axs[0].set_title('每个样本应有相同权重')
axs[1].bar(['正确按样本数','错误按批次数'],[2.8,2.5],color=[C[2],C[4]])
for i,v in enumerate([2.8,2.5]):axs[1].text(i,v+.05,str(v),ha='center')
axs[1].set_ylim(0,3.4);axs[1].set_title('(2×1 + 3×4) / 5 = 2.8');f.tight_layout();save(f,'05_reduction.png')
f,a=plt.subplots(figsize=(9,4));epochs=[int(r['epoch']) for r in h]
for key,label,col in [('train_online_mse','训练期间：变化的模型 + Dropout',C[1]),('train_eval_mse','轮末训练集：固定模型 + eval',C[0]),('validation_mse','轮末验证集：固定模型 + eval',C[2])]:a.plot(epochs,[float(r[key]) for r in h],'.-',label=label,color=col)
a.axvline(s['best_epoch'],color=C[3],ls='--',label='按验证选择 epoch 10');a.set_xlabel('完成的 epoch');a.set_ylabel('平均平方误差');a.set_title('三条曲线的测量协议不同');a.legend(fontsize=9);a.grid(alpha=.2);f.tight_layout();save(f,'06_history.png')
f,a=setup('从完整状态重建下一轮，而不是只读回权重',(9.2,4.2))
for x,y,txt,col in [(.1,3.2,'当前参数 + buffers',C[0]),(3.5,3.2,'动量缓冲 + 配置',C[1]),(6.9,3.2,'全局 RNG + 加载 RNG',C[3]),(.1,1.5,'数据摘要 + 拆分 ID',C[2]),(3.5,1.5,'轮次 + 历史 + 样本顺序',C[0]),(6.9,1.5,'最佳权重 + 最佳指标',C[1])]:box(a,x,y,2.9,1.1,txt,col)
a.text(5,.45,'校验 → 新建对象 → 装载模型/优化器 → 恢复 RNG → 从下一完整 epoch 开始',ha='center',fontsize=10)
save(f,'07_checkpoint.png')
f,a=plt.subplots(figsize=(9,3.8));labels=['完整恢复','缺动量','缺全局 RNG','缺加载 RNG'];values=[0]+[s['ablations'][k]['final_parameter_max_gap'] for k in ('optimizer','torch_rng','loader_rng')]
a.bar(labels,values,color=[C[2],C[1],C[3],C[4]])
for i,v in enumerate(values):a.text(i,v+.006,f'{v:.6f}',ha='center')
a.set_ylim(0,.39);a.set_ylabel('相对未中断轨迹的最终参数最大绝对差');a.set_title('第 4 轮边界中断，继续到第 12 轮');f.tight_layout();save(f,'08_resume.png')
f,a=plt.subplots(figsize=(7.5,4));p=json.loads((ROOT/'outputs/test-predictions.json').read_text());a.scatter([q['target'] for q in p],[q['prediction'] for q in p],color=C[0]);a.plot([-2,2],[-2,2],'--',color='#888');a.set_xlabel('合成目标 y');a.set_ylabel('选中 epoch 10 的预测');a.set_title('封存的 12 行教学测试集，MSE ≈ 0.139051');a.grid(alpha=.2);f.tight_layout();save(f,'09_test.png')
f,a=setup('模式与求导开关是两条不同轴',(8.5,3.9))
for x,y,txt,col in [(2,2.7,'训练模式 + 记录图\nDropout 随机；可反向',C[0]),(6,2.7,'训练模式 + no_grad\nDropout 仍随机',C[1]),(2,.8,'评估模式 + 记录图\nDropout 关闭；仍可反向',C[3]),(6,.8,'评估模式 + no_grad\n本讲的观测验证协议',C[2])]:box(a,x,y,3.6,1.35,txt,col)
a.text(.1,3.4,'train()',va='center');a.text(.1,1.5,'eval()',va='center');a.text(3.8,4.6,'grad enabled',ha='center');a.text(7.8,4.6,'no_grad',ha='center');save(f,'10_modes.png')
print('10 figures rebuilt after checking',len(EXPECTED),'fixed files')
