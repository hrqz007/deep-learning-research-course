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

def velocity(x, t, grid, coefficients):
    # 每个时间箱的线性回归给出 a(t), b(t)，插值避免时间箱边界跳跃。
    a = np.interp(t, grid, coefficients[:, 0])
    b = np.interp(t, grid, coefficients[:, 1])
    return a*x+b

def integrate(z, steps, grid, coefficients):
    x=z.copy(); h=1.0/steps
    for k in range(steps):
        # 显式 Euler；本讲故意保留一阶误差，以供步长消融。
        x += h*velocity(x, k*h, grid, coefficients)
    return x

def main():
    rng=np.random.default_rng(95)
    grid=np.linspace(0,1,51); rows=[]; training=[]
    mu=2.0; sigma=0.7
    for t in grid:
        # 独立配对：目标速度 x1-x0 含随机性；回归估计条件均值。
        x0=rng.normal(size=2000); x1=rng.normal(mu,sigma,2000)
        xt=(1-t)*x0+t*x1; target=x1-x0
        design=np.column_stack([xt,np.ones(len(xt))])
        coef=np.linalg.lstsq(design,target,rcond=None)[0]; rows.append(coef)
        training.append([t,float(np.mean((design@coef-target)**2))])
    coefficients=np.array(rows)
    # 独立测试随机数，不复用训练样本。
    z=rng.normal(size=10000); affine=mu+sigma*z
    metrics={"seed":95,"training_pairs":102000,"target_mean":mu,"target_std":sigma,"steps":{}}
    samples={"base":z,"affine":affine,"coefficients":coefficients,"grid":grid}
    for steps in (10,40,160):
        x=integrate(z,steps,grid,coefficients); samples[f"flow_{steps}"]=x
        metrics['steps'][str(steps)]={"mean":float(x.mean()),"std":float(x.std()),"paired_affine_rmse":float(np.sqrt(np.mean((x-affine)**2))),"central_95_coverage":float(np.mean(abs(x-mu)<1.96*sigma))}
    # 变量替换：log p_X(x)=log p_Z((x-mu)/sigma)-log sigma。
    logp=-0.5*z*z-0.5*np.log(2*np.pi)-np.log(sigma)
    samples['affine_log_density']=logp
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)); np.savez(OUT/'data.npz',**samples)
    np.savetxt(OUT/'training_loss.csv',np.array(training),delimiter=',',header='time,conditional_velocity_mse',comments='')
    exact_a=(-(1-grid)+grid*sigma**2)/((1-grid)**2+(grid*sigma)**2)
    exact_b=mu-exact_a*grid*mu
    fig,ax=plt.subplots(1,2,figsize=(9,3)); ax[0].plot(grid,coefficients[:,0],label='fitted a'); ax[0].plot(grid,exact_a,'--',label='oracle a'); ax[1].plot(grid,coefficients[:,1],label='fitted b'); ax[1].plot(grid,exact_b,'--',label='oracle b')
    for a in ax:a.set_xlabel('time');a.legend()
    savefig('01_velocity.png')
    plt.figure(figsize=(7,3));plt.hist(affine,bins=60,density=True,alpha=.5,label='exact affine');plt.hist(samples['flow_160'],bins=60,density=True,histtype='step',label='learned flow');plt.xlabel('x');plt.ylabel('density');plt.legend();savefig('02_distribution.png')
    plt.figure(figsize=(7,3));
    for start in np.linspace(-2,2,9):
        x=float(start);trace=[x]
        for k in range(160):x+=velocity(x,k/160,grid,coefficients)/160;trace.append(x)
        plt.plot(np.linspace(0,1,161),trace)
    plt.xlabel('time');plt.ylabel('transported position');savefig('03_paths.png')
    print(json.dumps(metrics,indent=2));return metrics
if __name__=='__main__':main()
