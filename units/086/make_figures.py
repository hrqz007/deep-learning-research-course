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
    flow('01_architecture.png', '读取数据与授予权限是两条不同路径', ['任务请求\n可信入口', '检索候选\n不可信文本', 'schema / 权限\n独立执行层', '受限结果\n数值与来源'], '本课仅规则控制器与mock工具，不运行真实语言模型或外部写操作。')
    flow('02_states.png', '状态机的主路径与停止条件', ['RECEIVED\n接收', 'RETRIEVED\n找证据', 'CALLING\n最多两次', 'FINISHED\n保存结果'], '旁路：缺证据 ABSTAINED；越权 DENIED；重试耗尽 EXHAUSTED。')
    flow('03_replay.png', '重放只读取过去的记录', ['原始执行\n工具有行为', '事件记录\n请求与结果', '哈希 / 版本\n一致性检查', '记录重放\n不重复执行'], '普通哈希检测意外变化，不认证记录作者；重放不同于当前环境重跑。')
    traces = json.loads((ROOT / 'outputs/traces.json').read_text())
    states = ['FINISHED', 'ABSTAINED', 'DENIED', 'EXHAUSTED']
    fig, ax = plt.subplots(figsize=(8, 3.3))
    ax.bar(states, [sum((t['events'][-1]['state'] == s for t in traces)) for s in states], color=COL[:4])
    ax.set(ylabel='任务数量', title='14个合成任务的实际终态')
    save(fig, '04_results.png')
if __name__ == '__main__':
    main()
