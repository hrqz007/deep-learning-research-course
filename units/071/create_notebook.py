"""Create the self-contained notebook; execute_notebook.py performs actual execution."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parent
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
cells=[
md('# DL071 分布外鲁棒性与不确定性\n\n本Notebook从零重训三个CPU模型，校准只使用校准集，所有测试域事先固定。输出另存notebook_outputs。图来自本次执行。'),
code('from pathlib import Path\nimport json, numpy as np, torch\nimport matplotlib.pyplot as plt\nfrom experiment import run, selective, model, logits, softmax\ntorch.set_num_threads(1)\nprint("PyTorch", torch.__version__, "device=cpu, threads=1")'),
md('## 1 先验证一个有明确分母的手算\n阈值0.75接受两条，其中一条错误。'),
code('p=np.array([[.9,.1],[.2,.8],[.7,.3],[.4,.6]])\ny=np.array([0,0,0,1])\nfor tau in [.75,0.,1.]:\n    print(tau, selective(p,y,tau))'),
md('## 2 实际训练与校准\n这里不是读取预置图；执行完整默认训练。'),
code('result=run("notebook_outputs")\nprint("Trained",len(result["runs"]),"models")\nfor r in result["runs"]:\n    print("seed",r["seed"],"temperature",r["temperature"],"threshold",r["threshold"])'),
md('## 3 逐域计数与空接受集\n输出null对应Python None，不能解释成零风险。'),
code('first=result["runs"][0]\nfor name,values in first["domains"].items():\n    s=values["selective"]\n    print(name,"accepted",s["accepted"],"coverage",s["coverage"],"risk",s["risk"])'),
md('## 4 重载全部权重\n只加载可信课程或自己生成的文件；这些是推理checkpoint。'),
code('root=Path("notebook_outputs")\ndata=np.load(root/"data.npz")\npred=np.load(root/"predictions.npz")\nfor r in result["runs"]:\n    ck=torch.load(root/f"model_seed{r[\'seed\']}.pt",weights_only=True)\n    m=model(); m.load_state_dict(ck["state_dict"])\n    q=softmax(logits(m,data["shortcut_flip_x"]),ck["temperature"])\n    np.testing.assert_array_equal(q,pred[f"seed{r[\'seed\']}_shortcut_flip_p"])\nprint("All three checkpoint predictions match exactly")'),
md('## 5 绘制本次执行的风险覆盖曲线\n全部测试阈值只用于描述曲线，不据此挑部署阈值。'),
code('fig,ax=plt.subplots(figsize=(8,4))\nfor name,v in first["domains"].items():\n    curve=v["risk_coverage"];ax.plot(curve["coverage"],curve["risk"],label=name)\nax.set(xlabel="Coverage",ylabel="Selective risk",ylim=(-.03,1.03))\nax.legend(ncol=2);fig.tight_layout();show(fig)'),
md('## 6 温度选择与置信但错误\n左图只用校准集，右图使用反转测试标签评估。'),
code('fig,axes=plt.subplots(1,2,figsize=(10,3.5))\nsearch=first["calibration_search"]\naxes[0].plot(search["temperatures"],search["nll"]);axes[0].set(xlabel="Temperature",ylabel="Calibration NLL")\nq=pred["seed17_shortcut_flip_p"];wrong=q.argmax(1)!=data["test_y"]\naxes[1].hist(q.max(1)[wrong],bins=20,color="tomato");axes[1].set(xlabel="Confidence on wrong predictions",ylabel="Count")\nfig.tight_layout();show(fig)\nprint(first["domains"]["shortcut_flip"]["confident_wrong_examples"][0])'),
md('## 7 与固定结果核对\n只比较确定性数值，耗时不应强求相同。'),
code('reference=json.loads(Path("outputs/results.json").read_text())\nfor a,b in zip(result["runs"],reference["runs"]):\n    np.testing.assert_allclose(a["losses"],b["losses"],rtol=1e-6,atol=1e-8)\n    for name in a["domains"]:\n        if a["domains"][name]["selective"]!=b["domains"][name]["selective"]:\n            raise RuntimeError("Selective metrics differ: "+name)\nprint("Deterministic losses and selective reports match reference")'),
md('## 结论边界\n同分布校准、低熵和成员一致都不保证分布外正确。这个实验未验证真实用户、自然图像或对抗攻击，也没有实现风险上界控制。')]
cells[1].source += '\nfrom io import BytesIO\nfrom IPython.display import Image, display\ndef show(fig):\n    buffer=BytesIO()\n    fig.savefig(buffer,format="png",dpi=140,bbox_inches="tight")\n    display(Image(data=buffer.getvalue()))\n    plt.close(fig)'
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
nbf.write(nb,ROOT/'experiment.ipynb')
