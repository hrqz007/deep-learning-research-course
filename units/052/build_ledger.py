"""Append the full numerical ledger to lecture; replace only its generated tail."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parent
MARK='<!-- generated-ledger -->'
def mat(a):
    a=np.asarray(a)
    if a.ndim==1:a=a[:,None]
    return r'\begin{pmatrix}'+r'\\'.join('&'.join(f'{v:.9f}' for v in row) for row in a)+r'\end{pmatrix}'
def pair(x):return ', '.join(f'{v:.9f}' for v in x)
def main():
    r=json.loads((ROOT/'outputs/results.json').read_text());text=(ROOT/'lecture.md').read_text().split(MARK)[0]+MARK+'\n'
    for s in r['hand']:
        t=s['step'];text+=f'\n### 状态 {t}\n\n总loss={s["loss"]:.12f}。两份样本自身的half-MSE为{s["sample_loss"][0]:.12f}、{s["sample_loss"][1]:.12f}，两者平均为总loss。\n\n|参数|第1行|第2行|\n|---|---:|---:|\n'
        for name,w in zip(['WQ','WK','WV'],s['weights']):text+=f'|{name}|{w[0][0]:.9f}|{w[1][0]:.9f}|\n'
        text+='\n|样本 位置|Q|K|V|O|目标|GO|\n|---|---:|---:|---:|---:|---:|---:|\n'
        for b in range(2):
            for i in range(2):text+='|'+ '|'.join([f'{b+1} {i+1}']+[f'{s[k][b][i][0]:.9f}' for k in ['q','k','v','prediction','target','go']])+'|\n'
        for b in range(2):
            text+=f'\n样本{b+1}的分数与权重：\n\n$$S_{b+1}={mat(s["score"][b])},\\quad A_{b+1}={mat(s["a"][b])}.$$\n\n'
            text+=f'该样本两个位置的完整Jacobian：\n\n$$J_1={mat(s["jacobian"][b][0])},\\quad J_2={mat(s["jacobian"][b][1])}.$$\n'
        text+='\n局部反向。GA和GS每格为一整行两项。\n\n|样本 位置|GA|GS|GQ|GK|GV|\n|---|---|---|---:|---:|---:|\n'
        for b in range(2):
            for i in range(2):text+='|'+'|'.join([f'{b+1} {i+1}',pair(s['ga'][b][i]),pair(s['gs'][b][i])]+[f'{s[k][b][i][0]:.9f}' for k in ['gq','gk','gv']])+'|\n'
        text+='\n位置已求和的样本贡献与总梯度：\n\n|参数坐标|样本1|样本2|总梯度|\n|---|---:|---:|---:|\n'
        for j,name in enumerate(['WQ','WK','WV']):
            for d in range(2):text+=f'|{name}[{d+1}]|{s["weight_gradient_per_sample"][j][0][d][0]:.9f}|{s["weight_gradient_per_sample"][j][1][d][0]:.9f}|{s["weight_gradient"][j][d][0]:.9f}|\n'
        text+='\n输入两特征的三条路径及总和：\n\n|样本 位置|Q路径|K路径|V路径|GX合计|\n|---|---|---|---|---|\n'
        for b in range(2):
            for i in range(2):text+='|'+'|'.join([f'{b+1} {i+1}']+[pair(s['input_gradient_paths'][j][b][i]) for j in range(3)]+[pair(s['input_gradient'][b][i])])+'|\n'
        if t<2:text+=f'\n全部梯度完成后，六个坐标同步减去0.1倍总梯度，进入状态{t+1}；本状态反向不能混用更新后的参数。\n'
    text+='\n## 一手来源\n\n[1] Vaswani等，Attention Is All You Need，2017。结构与缩放动机。https://arxiv.org/abs/1706.03762\n\n[2] PyTorch 2.7 SDPA官方文档。https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html\n\n[3] PyTorch 2.7 MultiheadAttention官方文档。https://docs.pytorch.org/docs/2.7/generated/torch.nn.MultiheadAttention.html\n\n[4] Jain与Wallace，Attention is not Explanation，NAACL 2019。https://aclanthology.org/N19-1357/\n\n2026-10-05核对。全文推导、数值样本、图与数据为原创；全屏蔽行行为来自真实运行。\n'
    (ROOT/'lecture.md').write_text(text)
if __name__=='__main__':main()
