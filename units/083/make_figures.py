"""Original figures from this lesson's actual outputs; no network or stock images."""
from pathlib import Path
import os, json
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', '/tmp/dl-course-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 160})
COL = ['#2563a8', '#e78737', '#479b85', '#915eaf', '#a85555', '#687587']

def save(fig, name):
    fig.savefig(ROOT / 'figures' / name, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def flow(name, title, labels, footer):
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    ax.set_title(title, pad=14, fontsize=14)
    n = len(labels)
    xs = np.linspace(0.2, 8.2, n)
    w = 1.6
    for j, (x, label) in enumerate(zip(xs, labels)):
        ax.add_patch(FancyBboxPatch((x, 1.1), w, 1.1, boxstyle='round,pad=.09', facecolor='#eef4fa', edgecolor=COL[j % len(COL)], lw=1.5))
        ax.text(x + w / 2, 1.65, label, ha='center', va='center', fontsize=11)
        if j < n - 1:
            ax.annotate('', xy=(xs[j + 1] - 0.08, 1.65), xytext=(x + w + 0.08, 1.65), arrowprops={'arrowstyle': '->', 'color': '#536477', 'lw': 1.5})
    ax.text(5, 0.42, footer, ha='center', va='center', fontsize=10, color='#536477')
    save(fig, name)

def main():
    (ROOT / 'figures').mkdir(exist_ok=True)
    m = json.loads((ROOT / 'outputs/metrics.json').read_text())
    u = int(ROOT.name)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis('off'); ax.set_title('冻结主路与低秩支路并行读取同一输入')
    nodes=[(1,2,'输入 h\nd=12'),(4.4,3,'冻结 W0h\n不更新'),(4.4,1,'支路 B(Ah)\nr=1/2/4'),(8,2,'两路相加\nlogits k=8')]
    for x,y,t in nodes: ax.text(x,y,t,ha='center',va='center',bbox={'boxstyle':'round,pad=.5','facecolor':'#eef4fa','edgecolor':COL[0]})
    for a,b in [(0,1),(0,2),(1,3),(2,3)]:
        x,y,_=nodes[a]; xx,yy,_=nodes[b]; ax.annotate('',xy=(xx-.75,yy),xytext=(x+.65,y),arrowprops={'arrowstyle':'->','color':'#536477','lw':1.6})
    ax.text(5,.12,'参数节省发生在可训练支路；原始权重仍参与计算。',ha='center',fontsize=10,color='#536477')
    save(fig, '01_flow.png')
    hist = json.loads((ROOT / 'outputs/history.json').read_text())
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for j, (k, v) in enumerate(hist.items()):
        a = np.asarray(v)
        ax.plot(a[:, 0], a[:, 1], label=k, color=COL[j])
    ax.set(xlabel='更新步', ylabel='训练 NLL (nat/样本)', title='同一数据与固定协议下的训练损失')
    ax.legend(ncol=3, fontsize=9)
    save(fig, '02_training.png')
    fig, ax = plt.subplots(figsize=(8, 3.4))
    keys = list(m['results'])
    x = np.arange(len(keys))
    ax.bar(x - 0.18, [m['results'][k]['adaptation_test']['nll'] for k in keys], 0.36, label='新领域测试', color=COL[0])
    ax.bar(x + 0.18, [m['results'][k]['retention_test']['nll'] for k in keys], 0.36, label='旧领域保留', color=COL[1])
    ax.set_xticks(x, keys)
    ax.set(ylabel='NLL (nat/样本)，越低越好', title='适配收益与遗忘必须一起看')
    ax.legend()
    save(fig, '03_tradeoff.png')
if __name__ == '__main__':
    main()
