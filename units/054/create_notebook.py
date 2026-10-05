"""Build a reproducible notebook; no external service needed."""
from pathlib import Path
import nbformat as n
root=Path(__file__).resolve().parent
cells=[n.v4.new_markdown_cell('''# DL054 预训练目标与数据构造\n\n同一原创语料的AR/MLM数据构造与独立文档packing。先预测结果再运行；这个未训练探针检查结构，不比较语言能力。'''),n.v4.new_code_cell('''from pathlib import Path
import sys,json,unittest
candidates=[Path.cwd(),Path.cwd()/"units"/"054"]
ROOT=next((p.resolve() for p in candidates if p.name=="054" and (p/"experiment.py").exists()),None)
if ROOT is None: raise RuntimeError("从054目录或课程根启动Notebook")
sys.path.insert(0,str(ROOT))
from experiment import *
from generate_data import VOCAB,corpus
setup();REPLAY=ROOT/"notebook_replay"
print("torch",torch.__version__,"CPU float64")'''),n.v4.new_markdown_cell('''## 1 用词元名称检查预测对\nA=[蓝,猫]，B=[红]。每个文档独立移位，然后拼接并重置位置。不要把IGNORE送入Embedding。'''),n.v4.new_code_cell('''batch=ar_packed([[4,7],[5]])
for k in ["x","y","position","segment","allow"]: print(k,batch[k])
print("inputs",[VOCAB[i] for i in batch["x"][0]])
print("targets",[VOCAB[i] for i in batch["y"][0]])'''),n.v4.new_markdown_cell('''## 2 手算概率 梯度与两个更新\n只优化独立logits，隔离NLL归约。位置2无有效标签，不参加分母。'''),n.v4.new_code_cell('''for row in nll_ledger():
    print(json.dumps(row,ensure_ascii=False,indent=2))'''),n.v4.new_markdown_cell('''## 3 两种目标和packing干预\nrun重新生成相同语料、构造全部数组、执行两层未训练Transformer，并保留错误标签/位置/跨文档对照。'''),n.v4.new_code_cell('''results=run(REPLAY)
print("AR vs MLM有效数",results["ar"]["count"],results["mlm"]["count"])
print("注意：两个平均NLL不是目标优劣排名")'''),n.v4.new_markdown_cell('''## 4 全部独立测试\n含解析梯度、全参数packing等价、忽略输出仍能提供上下文梯度、去重、边界与失败对照。'''),n.v4.new_code_cell('''from test_experiment import Tests
outcome=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
if not outcome.wasSuccessful(): raise RuntimeError("DL054 tests failed")'''),n.v4.new_markdown_cell('''## 5 重画并显示6幅图\n需要Noto Sans CJK，见lab.pdf。图读取本次重跑的结果，不从保存图片伪造执行。'''),n.v4.new_code_cell('''from make_figures import main as draw
from IPython.display import display,Image
folder=draw(REPLAY/"results.json",REPLAY/"figures")
for path in sorted(folder.glob("*.png")):
    print(path.name);display(Image(filename=str(path)))'''),n.v4.new_markdown_cell('''## 6 结论\n写出三种mask各自控制什么；解释EOS为什么不强制隔离文档；给出按有效token归约的分子与分母。保留所有失败对照，不把小语料未训练探针推广为真实中文质量结论。''')]
n.write(n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}),root/'experiment.ipynb')
