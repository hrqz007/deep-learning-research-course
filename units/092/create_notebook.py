from pathlib import Path
import nbformat as nbf
R=Path(__file__).resolve().parent
cells=[]
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))
md('# DL092 多模态研究的证据与偏差\n本Notebook从零重训15个模型到notebook_outputs，不覆盖随包outputs。阅读lecture.md理解标签与统计口径。CPU、合成数据、无需联网。')
code('from pathlib import Path\nimport json,numpy as np\nimport experiment as exp\nprint("NumPy",np.__version__)')
md('## 数据与标签\n检查真实潜变量仅用于生成和审计，不进入模型。')
code('x,y,g,a=exp.data(8,9201)\nprint("v, b, z:",x)\nprint("y:",y)\nassert np.array_equal(y,(a*x[:,1]>0).astype(float))')
md('## 手算与特征\n确认交互使用带噪视觉v，而非隐藏真值a。')
code('print(exp.features(x,"interaction"))\nprint("sigmoid(-1.2)=",exp.sigmoid(-1.2))')
md('## 完整真实训练\n运行五种方法、三种子、八个测试条件。')
code('result=exp.main("notebook_outputs")\nprint(json.dumps(result["aggregate"],indent=2))')
md('## 配对与重标\n相同输入可有不同评价问题。先明确标签规则再看准确率。')
code('ds=exp.domains(x,y,a)\nprint("original:",ds["permuted_original_label"][1])\nprint("relabelled:",ds["permuted_relabelled"][1])\nprint("delta",result["paired_delta"],"CI",result["paired_example_bootstrap_ci"])')
md('## 自动测试\n有限差分检查手写梯度，也检查反事实标签。')
code('import unittest\nsuite=unittest.defaultTestLoader.discover(".",pattern="test_experiment.py")\ncheck=unittest.TextTestRunner(verbosity=2).run(suite)\nif not check.wasSuccessful(): raise RuntimeError("test failure")')
md('## 直接从本次运行绘制诊断\n此图使用notebook_outputs对应的result，避免混用随包结果。')
code('import matplotlib.pyplot as plt\nfrom IPython.display import display, Image\nimport io\nfig,ax=plt.subplots(figsize=(8,3))\nrows=[r for r in result["aggregate"] if r["domain"]=="shortcut_flip"]\nax.bar([r["method"] for r in rows],[100*r["error_mean"] for r in rows])\nax.set_ylabel("Shortcut flip error (%)")\nax.tick_params(axis="x",rotation=20)\nfig.tight_layout()\nbuf=io.BytesIO()\nfig.savefig(buf,format="png",dpi=140,bbox_inches="tight")\ndisplay(Image(data=buf.getvalue()))\nplt.close(fig)')
md('## 写下结论\n记录一个支持的比较、一项原域代价、一项无法外推的结论。不要把三种子误称为三个独立现实数据集。')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}});nbf.write(nb,R/'experiment.ipynb')
