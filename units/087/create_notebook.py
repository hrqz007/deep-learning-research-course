"""Create the lesson notebook from local original cells; execution is separate."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parent
def main():
 cells=[]
 cells.append(nbf.v4.new_markdown_cell('# DL087 Agent与检索系统的研究评价\n\n本Notebook使用真实Python内核顺序运行。先读lecture.md与lab.md。只验证明确限定的离线教学机制，不代表真实LLM性能。'))
 cells.append(nbf.v4.new_code_cell('from pathlib import Path\nimport json, sys\nimport numpy as np\nimport experiment as exp\nprint("Python",sys.version.split()[0],"NumPy",np.__version__)'))
 cells.append(nbf.v4.new_markdown_cell('## 完整实验与真实输出\n运行会在本目录outputs写入可复核结果。'))
 cells.append(nbf.v4.new_code_cell('result=exp.main()\nprint(json.dumps(result,ensure_ascii=False,indent=2))'))
 cells.append(nbf.v4.new_markdown_cell('## 冻结任务与金标准隔离'))
 cells.append(nbf.v4.new_code_cell('docs,tasks=exp.fixture()\nt=tasks[20]\nvisible={k:t[k] for k in ["id","query"]}\nprint("visible",visible)\nprint("answer",exp.run_system(visible,docs))'))
 cells.append(nbf.v4.new_markdown_cell('## 证据标准的区别'))
 cells.append(nbf.v4.new_code_cell('t=tasks[0]\nr=exp.run_system({k:t[k] for k in ["id","query"]},docs,"no_citation")\nprint(exp.score(t,r))'))
 cells.append(nbf.v4.new_markdown_cell('## 配对重采样'))
 cells.append(nbf.v4.new_code_cell('print(exp.paired_bootstrap([1,0,1,0],[1,0,0,0]))\na=result["results"]["full"]["per_task_mean_success"]\nb=result["results"]["no_retry"]["per_task_mean_success"]\nprint(exp.paired_bootstrap(a,b))'))
 cells.append(nbf.v4.new_markdown_cell('## 全部单元测试\n同时检查边界与失败路径，不只观察损失下降。'))
 cells.append(nbf.v4.new_code_cell('import unittest\nsuite=unittest.defaultTestLoader.discover(str(Path.cwd()),pattern="test_experiment.py")\ncheck=unittest.TextTestRunner(verbosity=2).run(suite)\nif not check.wasSuccessful(): raise RuntimeError("tests failed")'))
 cells.append(nbf.v4.new_markdown_cell('## 从输出重建原创图\n图的数值来自本次实验，不是示意数字。'))
 cells.append(nbf.v4.new_code_cell('import make_figures\nmake_figures.main()\nfrom IPython.display import display, Image\nfigures=sorted(Path("figures").glob("*.png"))\nprint("figure count",len(figures))\ndisplay(Image(filename=str(figures[-1])))'))
 cells.append(nbf.v4.new_markdown_cell('## 研究记录\n写下一个受支持结论、一个失败/限制，以及下一步需要的新证据。参考答案在answers.md，不能用参考结论替代你自己的检查。'))
 cells.append(nbf.v4.new_code_cell('print("Output files:",sorted(p.name for p in Path("outputs").iterdir() if p.is_file()))'))
 nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12"}})
 nbf.write(nb,ROOT/"experiment.ipynb")
if __name__=="__main__":main()
