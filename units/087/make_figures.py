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
    flow('01_protocol.png', '测试题、金标准与控制器的隔离', ['冻结80题\n保存哈希', '仅 id/query\n交给控制器', '轨迹与结果\n三个种子', '独立评分\n读取expected'], '任务编号只用于固定故障随机数，不按编号返回答案。')
    flow('02_failure_tree.png', '从最终失败回溯证据链', ['未完成任务\n正式标准', '检索证据\n编号与别名', '工具恢复\n故障和预算', '引用支持\n数值与来源'], '“遇到故障”可以恢复；“没有工具异常”也可能读错记录。')
    keys = list(m['results'])
    labels = ['完整', '去规范化', '去重试', '去引用', '无检索']
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(x - 0.18, [m['results'][k]['success_rate'] for k in keys], 0.36, label='正式成功（含证据）', color=COL[0])
    ax.bar(x + 0.18, [m['results'][k]['value_only_score'] for k in keys], 0.36, label='仅数值诊断', color=COL[1])
    ax.set_xticks(x, labels)
    ax.set(ylim=(0, 1.12), ylabel='240次运行平均分', title='评价标准改变结论，必须事先冻结')
    ax.legend()
    save(fig, '03_ablation.png')
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for j, k in enumerate(keys):
        d = m['results'][k]
        ax.scatter(d['mean_simulated_cost'], d['success_rate'], color=COL[j], s=55)
        ax.annotate(labels[j], (d['mean_simulated_cost'], d['success_rate']), xytext=(6, 5), textcoords='offset points', fontsize=10)
    ax.set(xlabel='平均模拟成本单位（不是ms或美元）', ylabel='正式成功率', title='质量与实际使用的mock预算')
    ax.margins(0.25)
    save(fig, '04_frontier.png')
if __name__ == '__main__':
    main()
