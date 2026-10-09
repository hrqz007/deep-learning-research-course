"""Create a readable notebook; executing it really reruns the experiment."""
from pathlib import Path  # Work relative to the lesson, not a private machine path.
import nbformat as nbf  # Build a standard Notebook version 4 document.
root=Path(__file__).resolve().parent  # Locate the shared experiment implementation.
nb=nbf.v4.new_notebook()  # Start without stale results.
nb.cells=[
 nbf.v4.new_markdown_cell("# DL077 可解释性与因果主张\n\n对应原catalog T5。重启内核并按顺序运行；实验离线，全部数据本地生成。正文解释见lecture.md，练习见lab.md。"),
 nbf.v4.new_code_cell("# 引入共享的完整实验实现；无网络调用。\nfrom experiment import run\n# 另存结果，保留发行版outputs证据。\nresults = run('notebook_outputs')\n# 显示真实训练和评价的结果。\nresults"),
 nbf.v4.new_markdown_cell("## 数字数组与复核\nNPZ保存普通数值数组；关闭pickle并列出形状，确认样本与字段含义。"),
 nbf.v4.new_code_cell("# 读取本次Notebook真正产生的数据。\nimport numpy as np\ndata = np.load('notebook_outputs/data.npz', allow_pickle=False)\n# 显示每个字段的形状，结合data_dictionary.md阅读。\n{key: data[key].shape for key in data.files}"),
 nbf.v4.new_markdown_cell("## 自动数学核查\n运行数值梯度、结构机制或后验检查；失败会明确中止。"),
 nbf.v4.new_code_cell("# unittest断言在普通和优化解释器下都生效。\nimport unittest\nimport test_experiment\nsuite = unittest.defaultTestLoader.loadTestsFromModule(test_experiment)\nchecked = unittest.TextTestRunner(verbosity=2).run(suite)\nif not checked.wasSuccessful():\n    raise RuntimeError('Numerical checks failed')"),
 nbf.v4.new_markdown_cell("## 内嵌实验结果图\n以下图直接使用本Notebook新产生的数组，并未复制旧的显示输出。"),
 nbf.v4.new_code_cell("# 本地绘图并显示，无在线图片。\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nfrom IPython.display import display, Image\nfrom io import BytesIO\npred = np.load('notebook_outputs/predictions.npz', allow_pickle=False)\nfig, ax = plt.subplots(figsize=(8, 3))\nax.scatter(data['test_y'],pred['test_prediction'],s=8,alpha=.5)\nax.plot([-12,12],[-12,12],'--',color='black',label='identity')\nax.set(xlabel='observed Y',ylabel='predicted Y')\nax.legend()\nfig.tight_layout()\n# 将真实新图编码为PNG，让Notebook离线打开也可见。\nbuf = BytesIO()\nfig.savefig(buf, format='png', dpi=140)\ndisplay(Image(data=buf.getvalue()))\nplt.close(fig)"),
 nbf.v4.new_markdown_cell("## 自我检查\n把本次数字与answers.md比较。预测准确、数学实现正确和因果/域外主张是不同问题；请分别写出证据和限制。")]
nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
nbf.write(nb,root/'experiment.ipynb')  # Save editable notebook source.
