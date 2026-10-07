from pathlib import Path
import nbformat as n
R=Path(__file__).resolve().parent;M=n.v4.new_markdown_cell;C=n.v4.new_code_cell
cells=[M('# DL060 VAE与ELBO\n\n二维连续与8×8二值图像均实际训练。先做手算，再运行12个600步模型。CPU离线数据，输出notebook_replay，不覆盖参考。'),C('''from pathlib import Path
import sys,json,math
import numpy as np
import torch
ROOT=next((p.resolve() for p in [Path.cwd(),Path.cwd()/"units"/"060"] if p.name=="060" and (p/"experiment.py").exists()),None)
if ROOT is None:raise RuntimeError("请从060或课程根目录启动")
sys.path.insert(0,str(ROOT))
from experiment import *
print("Python",sys.version.split()[0],"torch",torch.__version__,"NumPy",np.__version__)
'''),M('## 1 观察两个任务\n图像是生成器抽出的真二值观测，不是普通灰度图。'),C('''d=make_data()
for k,v in d.items():print(k,v.shape,"range",(v.min(),v.max()))
print("image values",np.unique(d["image_train"]))
'''),M('## 2 KL与重参数化梯度\n手算KL1.306853,z=(2,-1),mu梯度(1,1),logvar梯度(.5,-.5)。'),C('''mu=torch.tensor([[1.,0.]],requires_grad=True)
lv=torch.tensor([[math.log(4),0.]],requires_grad=True)
eps=torch.tensor([[.5,-1.]])
z=reparameterize(mu,lv,eps)
print("KL",kl_standard(mu,lv).item(),"z",z.detach().numpy())
z.sum().backward();print("grad_mu",mu.grad,"grad_logvar",lv.grad)
'''),M('## 3 观测常数与归约\n检查二维Gaussian常数和64像素求和。'),C('''x=torch.zeros(1,2)
print("Gaussian D2 sigma1",reconstruction_nll(x,x,"xy",1).item(),math.log(2*math.pi))
x=torch.zeros(1,64)
print("Bernoulli D64",reconstruction_nll(x,x,"image").item(),64*math.log(2))
'''),M('## 4 完整参考检查\n12份模型全部逐测试样本重放16次MC，核对干预前缀。'),C('''from test_experiment import main as check
check(ROOT/"outputs")
'''),M('## 5 从头训练全部12个模型\n两个任务、两个种子、三个预声明beta方案；最后统一评价标准负ELBO。'),C('''result=run(ROOT/"notebook_replay")
check(ROOT/"notebook_replay")
print(json.dumps(result["protocol"],ensure_ascii=False,indent=2))
'''),M('## 6 同时报告重构KL与配对打乱\nbeta100是人为压力；图像恢复仍不及standard，不能省略。'),C('''for row in result["results"]:print(row)
reference=json.loads((ROOT/"outputs/results.json").read_text())
print("max negative ELBO replay delta",max(abs(a["negative_elbo"]-b["negative_elbo"]) for a,b in zip(result["results"],reference["results"])))
'''),M('## 7 命名图中的对象\n这里先显示后验mu解码均值，再显示先验均值与观测样本；都不同于MC ELBO。'),C('''from IPython.display import display,Image
display(Image(filename=str(ROOT/"figures/02_image_reconstruction.png")))
display(Image(filename=str(ROOT/"figures/06_image_samples.png")))
'''),M('## 8 写下结论\n完成十题，说明观测模型、nat/例、MC估计与诱发坍塌边界。不要把生成均值当真正样本，也不要把下界当精确似然。')]
n.write(n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}}),R/'experiment.ipynb')
