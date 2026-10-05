"""An executed notebook that really retrains the fixed 12 configurations on Run All."""
from pathlib import Path
import nbformat as n
root=Path(__file__).resolve().parent
cells=[n.v4.new_markdown_cell('''# DL055 小型语言模型训练与规模\n\n完整Run All重新训练固定12配置，每run80步、8960有效token；总107520。约一分钟但取决于本机性能，不自动扩大预算。CPU、无下载、无付费API。先读lab.pdf再运行。'''),n.v4.new_code_cell('''from pathlib import Path
import sys,json,math,unittest
candidates=[Path.cwd(),Path.cwd()/"units"/"055"]
ROOT=next((p.resolve() for p in candidates if p.name=="055" and (p/"experiment.py").exists()),None)
if ROOT is None: raise RuntimeError("从055或课程根启动Notebook")
sys.path.insert(0,str(ROOT))
from experiment import *
setup(5501);REPLAY=ROOT/"notebook_replay"
print("CPU float32",torch.__version__,"单线程")
print(json.dumps(protocol(),ensure_ascii=False,indent=2))'''),n.v4.new_markdown_cell('''## 1 手算参数 token与计算预算\n参数不是激活；累计有效token包括重复；FLOPs是主要矩阵乘加模型，不是实测时间。'''),n.v4.new_code_cell('''for d in [16,32]:
    model=TinyLM(d)
    print("D",d,"公式参数",parameter_formula(d),"实际参数",sum(p.numel() for p in model.parameters()))
    print("前向矩阵FLOPs",forward_matmul_flops(d),"80步训练估算",3*80*forward_matmul_flops(d))
print("每步",BATCH*L,"每run",80*BATCH*L,"全部12run",12*80*BATCH*L)
print("参数公式不是进程RSS")'''),n.v4.new_markdown_cell('''## 2 独立测试\n覆盖标签、规则、参数、FLOPs、因果性、前缀长度、真实两步AdamW及全部随包权重读取。测试另外执行8步小训练，不改变正式结果。'''),n.v4.new_code_cell('''from test_experiment import Tests
outcome=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
if not outcome.wasSuccessful():raise RuntimeError("DL055 tests failed")'''),n.v4.new_markdown_cell('''## 3 真正重新训练\n每配置在独立Python进程运行，以取得该worker的峰值RSS。协议先写入再训练；测试划分不评分。运行期间不会扩大宽度、步数或种子范围。'''),n.v4.new_code_cell('''results=run(REPLAY)
print("累计有效训练token",results["total_effective_training_tokens"])
print(json.loads((REPLAY/"runtime.json").read_text()))'''),n.v4.new_markdown_cell('''## 4 实际优化链与重放核对\n同环境固定种子比较验证NLL；计时与RSS允许变化，不要求字节复现。读取一条真实偏置的两步记录，按讲义复算裁剪和AdamW。'''),n.v4.new_code_cell('''from statistics import mean,stdev
original=json.loads((ROOT/"outputs/results.json").read_text())
for old,new in zip(original["runs"],results["runs"]):
    delta=abs(old["final_validation"]["nll"]-new["final_validation"]["nll"])
    print(new["config"]["name"],"验证NLL",new["final_validation"]["nll"],"与随包差",delta)
    if delta>1e-5: raise RuntimeError("numeric replay differs; inspect version and hardware")
real=json.loads((REPLAY/"runs/d16_n128_s5501/results.json").read_text())
for step in real["steps"][:2]:print(step["step"],step["optimizer_example"])
for d in [16,32]:
    for pool in [32,128]:
        values=[r["final_validation"]["nll"] for r in results["runs"] if r["config"]["d"]==d and r["config"]["pool"]==pool]
        print(d,pool,"mean",mean(values),"sample SD",stdev(values))'''),n.v4.new_markdown_cell('''## 5 从本次全部结果重新画图\n7张图均内嵌真实PNG。吞吐与RSS来自本次测量，可能与PDF的原始执行记录不同；科学数值对齐不要求机器负载相同。'''),n.v4.new_code_cell('''from make_figures import main as draw
from IPython.display import display,Image
folder=draw(REPLAY/"results.json",REPLAY/"figures")
for path in sorted(folder.glob("*.png")):
    print(path.name);display(Image(filename=str(path)))'''),n.v4.new_markdown_cell('''## 6 写出受证据限制的结论\n比较固定token与约2.4亿矩阵FLOPs两个口径；保留32文档大模型的验证退化；检查颜色2规则位置；说明GPU未测、数据为合成、仅3初始化、未拟合幂律。不能由本课外推前沿模型或中文能力。''')]
n.write(n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}),root/'experiment.ipynb')
