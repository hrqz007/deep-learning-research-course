"""原创 CPU 教学实验。运行 python experiment.py；无需网络、GPU 或模型下载。"""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dl-mpl")
import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
def savefig(name):
    plt.tight_layout(); plt.savefig(FIG / name, dpi=160); plt.close()

def score_cond(x,t,c):
    # 加性 Gaussian 噪声下，条件边缘仍为 Gaussian。
    return -(x-2*c)/(0.36+t)
def score_uncond(x,t):
    var=.36+t
    # sigmoid(4*x/var) 是正类责任概率。裁剪只用于数值稳定。
    r=1/(1+np.exp(np.clip(-4*x/var,-60,60)))
    return (2*(2*r-1)-x)/var

def sample(seed,w,steps=400,n=4000):
    rng=np.random.default_rng(seed); T=9.;h=T/steps
    # 从无条件 p_T 精确采样。对 w=1，它并非条件 p_T，需声明起点偏差。
    x=2*rng.choice([-1,1],n)+np.sqrt(.36+T)*rng.normal(size=n)
    for k in range(steps):
        t=T-k*h
        su=score_uncond(x,t);sc=score_cond(x,t,1)
        guided=su+w*(sc-su)
        # 反向时钟 tau 增大，漂移是 +score；噪声方差为 h。
        x += h*guided+np.sqrt(h)*rng.normal(size=n)
    return x

def main():
    rng=np.random.default_rng(96);z=rng.normal(2,.6,4000);residual=rng.normal(0,.3,4000)
    observation=np.column_stack([z,.5*z+residual]);reconstruction=np.column_stack([z,.5*z])
    metrics={"target_label":1,"seed":96,"steps":400,"start_time":9,"compression_mse_per_coordinate":float(np.mean((observation-reconstruction)**2)),"guidance":{}}
    data={"observation":observation,"latent":z,"reconstruction":reconstruction}
    for w in (0,1,2,4):
        x=sample(96,w);data[f'w{w}']=x
        metrics['guidance'][str(w)]={"sign_consistency":float(np.mean(x>0)),"mean":float(x.mean()),"std":float(x.std()),"within_target_95":float(np.mean(abs(x-2)<1.96*.6)),"left_tail_coverage":float(np.mean(x<2-1.645*.6)),"right_tail_coverage":float(np.mean(x>2+1.645*.6))}
    np.savez(OUT/'data.npz',**data);(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
    fig,ax=plt.subplots(1,2,figsize=(9,3));ax[0].scatter(observation[:300,0],observation[:300,1],s=5,alpha=.4,label='observed');ax[0].plot(z[:300],.5*z[:300],'.',ms=2,label='decoded');ax[0].set_xlabel('x1');ax[0].set_ylabel('x2');ax[0].legend();ax[1].hist(residual,bins=40);ax[1].set_xlabel('discarded residual');savefig('01_compression.png')
    plt.figure(figsize=(7,3));
    for w in (0,1,2,4):plt.hist(data[f'w{w}'],bins=70,density=True,histtype='step',label=f'w={w}')
    plt.xlabel('generated latent');plt.ylabel('density');plt.legend();savefig('02_guidance.png')
    x=np.linspace(-4,4,300);plt.figure(figsize=(7,3));
    for w in (0,1,2,4):plt.plot(x,score_uncond(x,1)+w*(score_cond(x,1,1)-score_uncond(x,1)),label=f'w={w}')
    plt.xlabel('noisy latent');plt.ylabel('guided score at t=1');plt.legend();savefig('03_score.png')
    # 两条路径示意：强调条件进入去噪器，而编码/解码控制表示误差。
    fig,ax=plt.subplots(figsize=(9,3.4));ax.axis('off')
    for y,labels in [(0.72,['x','add noise','denoise + c','x sample']),(.25,['x -> E','latent z + noise','guided score + c','D -> x sample'])]:
        positions=[.07,.34,.62,.9]
        for x,label in zip(positions,labels):
            ax.text(x,y,label,ha='center',va='center',bbox=dict(boxstyle='round',fc='#eaf3fb',ec='#2675a5'),fontsize=10,transform=ax.transAxes)
        for a,b in zip(positions[:-1],positions[1:]):
            ax.annotate('',xy=(b-.10,y),xytext=(a+.08,y),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#555'))
    ax.text(0,.98,'Observation-space diffusion',transform=ax.transAxes)
    ax.text(0,.49,'Latent-space diffusion: compression and guidance are separate',transform=ax.transAxes)
    savefig('04_paths.png')
    print(json.dumps(metrics,indent=2));return metrics
if __name__=='__main__':main()
