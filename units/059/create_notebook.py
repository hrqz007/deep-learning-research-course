from pathlib import Path
import nbformat as n
R=Path(__file__).resolve().parent;M=n.v4.new_markdown_cell;C=n.v4.new_code_cell
cells=[M('# DL059 自监督与对比学习\n\n先写增强的预测，再从头运行。所有数据原创合成，CPU离线。Notebook重新训练所有配置，不覆盖参考outputs。'),C('''from pathlib import Path
import sys,json,math
import numpy as np
import torch
ROOT=next((p.resolve() for p in [Path.cwd(),Path.cwd()/"units"/"059"] if p.name=="059" and (p/"experiment.py").exists()),None)
if ROOT is None: raise RuntimeError("请从059或课程根目录启动")
sys.path.insert(0,str(ROOT))
from experiment import *
print("Python",sys.version.split()[0],"torch",torch.__version__,"NumPy",np.__version__)
'''),M('## 1 原始信号与独立干扰\n各划分独立，预训练不读验证或测试输入。'),C('''d=make_data()
for s in ["train","validation","test"]:print(s,d["x_"+s].shape,np.unique(d["y_"+s],return_counts=True))
'''),M('## 2 四视图手算\nτ=1预期0.551445；三个常量嵌入预期log5。'),C('''x=torch.eye(2,requires_grad=True)
loss=info_nce(x,x,1.);loss.backward()
print("loss",loss.item(),"hand",math.log(math.e+2)-1,"gradient",x.grad)
print("collapsed B3",info_nce(torch.ones(3,2),torch.ones(3,2)).item(),math.log(5))
'''),M('## 3 视图究竟改变了谁\n观察前后四维变化，说明正对索引不等于语义正确。'),C('''x=torch.from_numpy(d["x_train"][:128])
for kind in ["good","bad"]:
 v=augment(x,kind,torch.Generator().manual_seed(4))
 print(kind,"signal MSE",(v[:,:4]-x[:,:4]).square().mean().item(),"nuisance MSE",(v[:,4:]-x[:,4:]).square().mean().item())
'''),M('## 4 验证参考证据\n测试必须找到12组权重和探测器，并逐样本重放。'),C('''from test_experiment import main as check
check(ROOT/"outputs")
'''),M('## 5 真正完整重训\n九次350步训练与三个随机基准。写入独立notebook_replay。'),C('''result=run(ROOT/"notebook_replay")
check(ROOT/"notebook_replay")
print(json.dumps(result["protocol"],ensure_ascii=False,indent=2))
'''),M('## 6 汇报全部种子\n监督预训练有768标签，不能冒充与自监督相同标签预算。'),C('''for method in METHODS:
 vals=[r["test_accuracy"] for r in result["results"] if r["method"]==method]
 print(method,vals,"mean",np.mean(vals),"range",(min(vals),max(vals)))
reference=json.loads((ROOT/"outputs/results.json").read_text())
print("max replay accuracy delta",max(abs(a["test_accuracy"]-b["test_accuracy"]) for a,b in zip(result["results"],reference["results"])))
'''),M('## 7 从图形回到数字\n图形只显示一个初始化；定量结论包含全部三个。'),C('''from IPython.display import display,Image
display(Image(filename=str(ROOT/"figures/03_probe.png")))
'''),M('## 8 写下自己的解释\n回答lecture末尾十题，先解释目标和评价是否一致，再比较均值。只有一个固定合成划分，不能外推真实场景。')]
nb=n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}});n.write(nb,R/'experiment.ipynb')
