"""Rebuild the unexecuted learning notebook; execute_notebook.py runs it offline."""
from pathlib import Path
import nbformat as n
root=Path(__file__).resolve().parent
cells=[n.v4.new_markdown_cell('''# DL053 Transformer块与位置\n\n先读lecture.pdf，完成lab.pdf练习，再Run All。代码重算6配置对齐、两个SGD更新与7幅图，不下载数据。每一节都说明shape与观察范围。'''),
n.v4.new_code_cell('''from pathlib import Path
import sys, json
# Jupyter启动目录可能是课程根或单元目录；寻找本单元而非误导入其他课。
candidates=[Path.cwd(),Path.cwd()/"units"/"053"]
ROOT=next((p.resolve() for p in candidates if (p/"experiment.py").exists() and p.name=="053"),None)
if ROOT is None:
    raise RuntimeError("请从units/053或课程根目录启动Notebook")
sys.path.insert(0,str(ROOT))
from experiment import Block, Norm, hand_ledger, run, setup, sinusoidal, rotate_pairs
import torch
setup()
REPLAY=ROOT/"notebook_replay"
print("torch",torch.__version__,"CPU float64 单线程")'''),
n.v4.new_markdown_cell('''## 1 拆头之前先标号\n原shape为B=1,L=3,D=8。打印每头位置，合头要先把H、L轴换回。'''),
n.v4.new_code_cell('''x=torch.arange(24).reshape(1,3,8)
heads=x.reshape(1,3,2,4).transpose(1,2)
print(heads)
print("正确合头",torch.equal(heads.transpose(1,2).reshape_as(x),x))
print("错误合头",torch.equal(heads.reshape_as(x),x))'''),
n.v4.new_markdown_cell('''## 2 完整前向与两步同步SGD\n每一步打印两次LN和残差后的输出。loss是四个标量的half-MSE；观察总体下降是否意味着每个坐标都改进。'''),
n.v4.new_code_cell('''ledger=hand_ledger()
for record in ledger:
    print("step",record["step"],"loss",record["loss"])
    print("output",record["out"])
print("首次fc2梯度",ledger[0]["gradients"]["fc2.weight"])
print("首次output梯度",ledger[0]["output_gradient"])
print(json.dumps(ledger[0]["backward_trace"],indent=2))
print("显式链式法则对齐误差",[r["manual_autograd_max"] for r in ledger])'''),
n.v4.new_markdown_cell('''## 3 官方参考及结构干预\nrun重新复制参数、分别反向传播，并重算排列、位置、未来、长度与RoPE检查。对齐不是学习质量评测。'''),
n.v4.new_code_cell('''results=run(REPLAY)
print(json.dumps(results["alignment"],indent=2))'''),
n.v4.new_markdown_cell('''## 4 独立测试\nunittest在普通模式和python -O中均有效。这里重跑全部6类测试，不用本单元保存的结果代替计算。'''),
n.v4.new_code_cell('''import unittest
from test_experiment import Tests
suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
outcome=unittest.TextTestRunner(verbosity=2).run(suite)
if not outcome.wasSuccessful():
    raise RuntimeError("DL053 tests failed")'''),
n.v4.new_markdown_cell('''## 5 重新画图并嵌入实际图片\n图由本次results.json和公式重绘。需要Noto Sans CJK；缺字体时按lab.pdf安装并设置DL_CJK_FONT。'''),
n.v4.new_code_cell('''from make_figures import main as draw
from IPython.display import display, Image
directory=draw(REPLAY/"results.json",REPLAY/"figures")
for path in sorted(directory.glob("*.png")):
    print(path.name)
    display(Image(filename=str(path)))'''),
n.v4.new_markdown_cell('''## 6 写下证据边界\n请回答：为什么双向长度变化不应强求零差？为什么pre/post对齐都成功不表示性能相同？为什么RoPE共同平移不变不证明无限上下文？答案见answers.pdf。实测为CPU float64、零dropout、有限小张量；未测试GPU、半精度、浏览器UI或任何自然语言能力。''')]
nb=n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
n.write(nb,root/'experiment.ipynb')
