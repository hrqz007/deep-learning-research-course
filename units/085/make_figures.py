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
    flow('01_pipeline.png', '偏好训练与独立事实评价', ['候选对\n同一问题', '偏好标准\n选择标签', '策略训练\nSFT / DPO', '独立测试\n冲突与事实'], '训练裁判的偏好不是事实真值；参考模型固定，不随当前策略更新。')
    keys = list(m['results'])
    labels = ['参考', 'SFT', 'DPO .1', 'DPO .5', 'DPO 2', '偏置裁判']
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for j, (cond, label) in enumerate([('id', '同分布'), ('balanced', '平衡'), ('conflict', '冲突')]):
        ax.bar(x + (j - 1) * 0.25, [m['results'][k]['evaluation'][cond]['sampled_correctness'] for k in keys], 0.25, label=label, color=COL[j])
    ax.set_xticks(x, labels)
    ax.set(ylim=(0, 1.07), ylabel='策略采样正确概率', title='独立事实评价暴露偏好捷径')
    ax.legend(ncol=3)
    save(fig, '02_accuracy.png')
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(x - 0.18, [m['results'][k]['weights'][0] for k in keys], 0.36, label='正确性权重', color=COL[0])
    ax.bar(x + 0.18, [m['results'][k]['weights'][1] for k in keys], 0.36, label='冗长权重', color=COL[1])
    ax.set_xticks(x, labels)
    ax.axhline(0, color='#999', lw=0.6)
    ax.set(ylabel='线性策略权重', title='偏置裁判把策略推向冗长特征')
    ax.legend()
    save(fig, '03_weights.png')
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for j, k in enumerate(keys):
        d = m['results'][k]['evaluation']['balanced']
        ax.scatter(d['mean_reference_kl'], d['sampled_correctness'], color=COL[j], s=45)
        ax.annotate(labels[j], (d['mean_reference_kl'], d['sampled_correctness']), xytext=(4, 5 if j % 2 else -13), textcoords='offset points', fontsize=9)
    ax.set(xlabel='相对参考平均 KL (nat)', ylabel='平衡测试采样正确概率', title='偏离幅度与事实质量不是同一个指标')
    ax.margins(0.2)
    save(fig, '04_kl.png')
if __name__ == '__main__':
    main()
