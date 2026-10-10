from pathlib import Path
import nbformat as n
R=Path(__file__).resolve().parent
cells=[]
def md(s):cells.append(n.v4.new_markdown_cell(s))
def code(s):cells.append(n.v4.new_code_cell(s))
md('# DL081 可运行实验：PINN与差分\n完整套件13项测试，8个代码单元；局部演示先跑子集，最后运行全部13项。\n本Notebook在本讲目录运行。所有参数真实训练，解析解只用于评价。先阅读lecture.md和lab.md。')
code('import numpy as np, json\nfrom pathlib import Path\nimport experiment as ex\nprint("NumPy",np.__version__)\nprint("工作目录",Path.cwd())')
md('## 1 手算的函数与离散基线\n确认sin(πx)满足方程；这里尚未训练网络。差分基线使用线性复杂度Thomas算法。')
code('x,u=ex.finite_difference(31)\nprint("节点最大误差",np.max(np.abs(u-np.sin(np.pi*x))))\nprint("端点",u[[0,-1]])')
md('## 2 检查计算梯度\n输入导数和参数梯度是不同对象，测试都必须通过。')
code('import unittest, test_experiment\nsuite=unittest.TestSuite()\nfor name in ["test_gradient","test_input_derivative","test_difference_equations"]:\n    suite.addTest(test_experiment.Tests(name))\nresult=unittest.TextTestRunner(verbosity=2).run(suite)\nassert result.wasSuccessful()')
md('## 3 完整真实训练\n两种初始化、五点消融和三个差分网格全部执行。训练将更新outputs与data/evaluation.csv；耗时因机器而异。')
code('metrics=ex.run()\nprint(json.dumps(metrics,indent=2,ensure_ascii=False))')
md('## 4 比较同口径指标\n训练残差零不是连续域残差零；差分离散守恒不能直接冒充连续守恒。')
code('for name in ["pinn","pinn_seed82","pinn_sparse"]:\n    m=metrics[name]\n    print(name,"相对L2",m["relative_l2"],"训练残差",m["train_residual_rmse"],"评价残差",m["heldout_residual_rmse"])\nprint("差分63两种守恒",metrics["fd_63"])')
md('## 5 加密独立评价\n4001点只改变观察分辨率，不修改参数；与1001点结果比较。')
code('z=np.load("outputs/pinn_sparse_weights.npz")\ntheta=[z[k] for k in ["w","b","v","c"]]\nx=np.linspace(0,1,4001)\nu,du,ddu=ex.field(theta,x)\nprint("4001点残差RMSE",np.sqrt(np.mean((-ddu-np.pi**2*np.sin(np.pi*x))**2)))')
md('## 6 重新生成原创图示\n图使用本次真实输出；可直接查看figures目录。')
code('import runpy\nrunpy.run_path("make_figures.py",run_name="__main__")\nfrom IPython.display import Image,display\ndisplay(Image(filename="figures/03_residual.png"))')
md('## 7 最终自检与反思\n说明该网络为什么仍然只是固定热源的一个解，不是热源函数到解函数的算子。')
code('result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_experiment))\nassert result.wasSuccessful()')
nb=n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}});n.write(nb,R/'experiment.ipynb')
