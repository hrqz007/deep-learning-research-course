"""Construct a notebook that actually reruns the entire research protocol."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parent
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
cells=[
md('# DL072 独立研究项目与审查\n\n本Notebook从零训练五方法×三个种子的完整项目。全部数据离线生成，结果另存notebook_outputs，图从本次结果生成。'),
code('from pathlib import Path\nimport json, numpy as np, torch\nimport matplotlib.pyplot as plt\nfrom experiment import run, network, predict, metrics, VARIANTS, DOMAINS\ntorch.set_num_threads(1)\nprint("PyTorch",torch.__version__,"CPU one thread")'),
md('## 1 读取预定研究协议\n主指标与辅助指标在观察测试结果前固定。'),
code('plan=json.loads(Path("research_plan.json").read_text())\nprint(plan["question"])\nprint(plan["primary"])\nprint(plan["variants"],plan["seeds"])'),
md('## 2 真实训练15个模型\n没有读取已有权重跳过训练。'),
code('result=run("notebook_outputs")\nprint("Models",len(result["runs"]),"updates",result["budget"]["optimizer_updates"])\nfor name,a in result["aggregate"].items():\n    print(name,{domain:round(v["mean_error"],6) for domain,v in a["domains"].items()})'),
md('## 3 检查配对输入\n固定一个样本，角标变动但中心数据不变。'),
code('root=Path("notebook_outputs");data=np.load(root/"data.npz")\nfig,axes=plt.subplots(1,3,figsize=(8,3))\nfor ax,name in zip(axes,["id","flip","blank"]):\n    ax.imshow(data[name+"_x"][0,0],cmap="RdBu_r",vmin=-2.2,vmax=2.2);ax.set_title(name);ax.axis("off")\nfig.tight_layout();show(fig)'),
md('## 4 重载所有checkpoint并复核预测\n方法预处理必须与训练定义配套。'),
code('saved=np.load(root/"predictions.npz")\nfor path in sorted(root.glob("*_seed*.pt")):\n    ck=torch.load(path,weights_only=True);m=network(ck["variant"]);m.load_state_dict(ck["state_dict"])\n    p=predict(m,data["flip_x"],ck["variant"])\n    np.testing.assert_array_equal(p,saved[f"{ck[\'variant\']}_seed{ck[\'seed\']}_flip"])\nprint("15 checkpoint predictions match exactly")'),
md('## 5 全方法四域均值和标准差\n误差条是三个固定数据训练种子的样本标准差，不是总体风险置信区间。'),
code('fig,ax=plt.subplots(figsize=(10,4))\nx=np.arange(len(DOMAINS));width=.16\nfor j,v in enumerate(VARIANTS):\n    a=result["aggregate"][v]["domains"]\n    ax.bar(x+(j-2)*width,[a[d]["mean_error"] for d in DOMAINS],width,yerr=[a[d]["sd_error"] for d in DOMAINS],capsize=3,label=v)\nax.set_xticks(x,DOMAINS);ax.set_ylabel("Error rate");ax.legend(ncol=3);fig.tight_layout();show(fig)'),
md('## 6 配对改善与原域代价\n先同种子相减，再统计差值；不要选最佳种子。'),
code('delta=np.array(result["paired_difference"]["by_seed"])\nprint("Paired differences",delta,"mean",delta.mean(),"sample SD",delta.std(ddof=1))\na=result["aggregate"]\nprint("ID error cost",a["random_corner"]["domains"]["id"]["mean_error"]-a["erm"]["domains"]["id"]["mean_error"])\nprint("Budget",result["budget"])'),
md('## 7 确认本次重跑与固定证据一致\n计时随环境负载变化，故不比较计时。'),
code('ref=json.loads(Path("outputs/results.json").read_text())\nfor a,b in zip(result["runs"],ref["runs"]):\n    if a["variant"]!=b["variant"] or a["seed"]!=b["seed"]: raise RuntimeError("Run order changed")\n    np.testing.assert_allclose(a["losses"],b["losses"],rtol=1e-6,atol=1e-8)\n    for domain in DOMAINS:\n        if a["domains"][domain]["errors"]!=b["domains"][domain]["errors"]: raise RuntimeError("Error counts changed")\nprint("All deterministic loss trajectories and error counts match")'),
md('## 最终审查\n随机化在已知捷径反转域改善，但原域与核心噪声域代价明显，简单遮挡表现相近。真实图像、未知位置、更多数据划分与风险控制仍待验证。本项目不提出发表价值或方法创新声明。')]
cells[1].source += '\nfrom io import BytesIO\nfrom IPython.display import Image, display\ndef show(fig):\n    buffer=BytesIO()\n    fig.savefig(buffer,format="png",dpi=140,bbox_inches="tight")\n    display(Image(data=buffer.getvalue()))\n    plt.close(fig)'
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
nbf.write(nb,ROOT/'experiment.ipynb')
