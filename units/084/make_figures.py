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
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_title('两步 MDP：四条可枚举轨迹')
    positions = [(0.8, 2, '起点\nθ0'), (3.6, 3, '左支\nθ左'), (3.6, 1, '右支\nθ右'), (7.9, 3.5, 'a1=0：G0=0.09'), (7.9, 2.5, 'a1=1：G0=0.36'), (7.9, 1.5, 'a1=0：G0=-0.1'), (7.9, 0.5, 'a1=1：G0=0.8')]
    for x, y, t in positions:
        ax.text(x, y, t, ha='center', va='center', bbox={'boxstyle': 'round', 'facecolor': '#eef4fa', 'edgecolor': COL[0]})
    for a, b, label in [(0, 1, 'a0=0'), (0, 2, 'a0=1，代价0.1'), (1, 3, ''), (1, 4, ''), (2, 5, ''), (2, 6, '')]:
        x, y, _ = positions[a]
        xx, yy, _ = positions[b]
        ax.annotate('', xy=(xx - 0.8, yy), xytext=(x + 0.45, y), arrowprops={'arrowstyle': '->', 'color': '#536477'})
        if label:
            ax.text((x + xx) / 2, (y + yy) / 2 + 0.2, label, fontsize=9)
    save(fig, '01_mdp.png')
    d = m['mdp']
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8, 3.3))
    ax.plot(x, d['exact_gradient'], 'o', label='精确枚举', color=COL[0])
    ax.errorbar(x + 0.07, d['mc_gradient'], yerr=np.array(d['mc_standard_error']) * 2, fmt='s', capsize=4, label='MC均值 ± 2标准误', color=COL[1])
    ax.set_xticks(x, ['起点参数', '左支参数', '右支参数'])
    ax.set(ylabel='收益梯度', title='50,000条随机轨迹与解析参考')
    ax.legend()
    save(fig, '02_gradient.png')
    fig, ax = plt.subplots(figsize=(8, 3.3))
    ax.bar(x - 0.18, d['variance_no_baseline'], 0.36, label='不减基线', color=COL[0])
    ax.bar(x + 0.18, d['variance_constant_baseline'], 0.36, label='减常数 J', color=COL[1])
    ax.set_xticks(x, ['起点参数', '左支参数', '右支参数'])
    ax.set(ylabel='单样本梯度方差', title='相同轨迹上的基线配对对照')
    ax.legend()
    save(fig, '03_variance.png')
    h = np.array(json.loads((ROOT / 'outputs/history.json').read_text()))
    fig, ax = plt.subplots(figsize=(8, 3.3))
    ax.plot(h[:, 0], h[:, 1], color=COL[0], label='精确梯度上升')
    ax.axhline(0.8, color=COL[1], ls='--', label='最佳确定轨迹')
    ax.set(xlabel='更新次数', ylabel='精确平均回报', title='训练曲线不含Monte Carlo梯度噪声')
    ax.legend()
    save(fig, '04_learning.png')
if __name__ == '__main__':
    main()
