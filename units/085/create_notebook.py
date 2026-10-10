"""Create the lesson notebook from local original cells; execution is separate."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parent
def main():
 cells=[]
 cells.append(nbf.v4.new_markdown_cell('# DL085 偏好学习与语言模型后训练\n\n本Notebook使用真实Python内核顺序运行。先读lecture.md与lab.md。只验证明确限定的离线教学机制，不代表真实LLM性能。'))
 cells.append(nbf.v4.new_code_cell('from pathlib import Path\nimport json, sys\nimport numpy as np\nimport experiment as exp\nprint("Python",sys.version.split()[0],"NumPy",np.__version__)'))
 cells.append(nbf.v4.new_markdown_cell('## 完整实验与真实输出\n运行会在本目录outputs写入可复核结果。'))
 cells.append(nbf.v4.new_code_cell('result=exp.main()\nprint(json.dumps(result,ensure_ascii=False,indent=2))'))
 cells.append(nbf.v4.new_markdown_cell('## 标签与相关性'))
 cells.append(nbf.v4.new_code_cell('X,y=exp.fixture()\nprint("train shape",X.shape,"same feature sign",np.mean(X[:,0]==X[:,1]))'))
 cells.append(nbf.v4.new_markdown_cell('## 参考起点与DPO'))
 cells.append(nbf.v4.new_code_cell('reference=np.array([.8,0.])\nprint("initial DPO loss",exp.objective(reference,X,y)[0],"log2",np.log(2))\nnp.testing.assert_allclose(exp.objective(reference,X,y)[0],np.log(2))'))
 cells.append(nbf.v4.new_markdown_cell('## 独立冲突测试'))
 cells.append(nbf.v4.new_code_cell('Xc,yc=exp.fixture(1000,0,853)\nw,_=exp.train(X,(X[:,1]>0).astype(int))\nprint("biased weights",w)\nprint("conflict evaluation",exp.evaluate(w,Xc,yc))'))
 cells.append(nbf.v4.new_markdown_cell('## 全部单元测试\n同时检查边界与失败路径，不只观察损失下降。'))
 cells.append(nbf.v4.new_code_cell('import unittest\nsuite=unittest.defaultTestLoader.discover(str(Path.cwd()),pattern="test_experiment.py")\ncheck=unittest.TextTestRunner(verbosity=2).run(suite)\nif not check.wasSuccessful(): raise RuntimeError("tests failed")'))
 cells.append(nbf.v4.new_markdown_cell('## 从输出重建原创图\n图的数值来自本次实验，不是示意数字。'))
 cells.append(nbf.v4.new_code_cell('import make_figures\nmake_figures.main()\nfrom IPython.display import display, Image\nfigures=sorted(Path("figures").glob("*.png"))\nprint("figure count",len(figures))\ndisplay(Image(filename=str(figures[-1])))'))
 cells.append(nbf.v4.new_markdown_cell('## 研究记录\n写下一个受支持结论、一个失败/限制，以及下一步需要的新证据。参考答案在answers.md，不能用参考结论替代你自己的检查。'))
 cells.append(nbf.v4.new_code_cell('print("Output files:",sorted(p.name for p in Path("outputs").iterdir() if p.is_file()))'))
 nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12"}})
 nbf.write(nb,ROOT/"experiment.ipynb")
if __name__=="__main__":main()
