"""Small CPU width study. All settings fixed before the held-out width is evaluated."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/dl-course-mpl')
import json, csv, time, platform
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'; FIG=ROOT/'figures'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
WIDTHS=[4,8,16,32,64]; SEEDS=[11,23,37,51,73]; STEPS=500

def target(x):
    return np.sin(2*x[:,0])+0.5*x[:,1]*x[:,2]+0.3*np.cos(3*x[:,3])

def dataset(seed,n):
    r=np.random.default_rng(seed); x=r.uniform(-1,1,(n,4)).astype('float32')
    y=(target(x)+r.normal(0,.1,n)).astype('float32')[:,None]
    return x,y

def fit_curve(params,loss):
    slope,intercept=np.polyfit(np.log(params),np.log(loss),1)
    return float(intercept),float(slope)

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    data={}
    for name,seed,n in [('train',101,1024),('validation',202,512),('test',303,1024)]:
        data[name+'_x'],data[name+'_y']=dataset(seed,n)
    np.savez(OUT/'data.npz',**data)
    tx,ty=map(torch.from_numpy,(data['train_x'],data['train_y']))
    vx,vy=map(torch.from_numpy,(data['validation_x'],data['validation_y']))
    xx,yy=map(torch.from_numpy,(data['test_x'],data['test_y']))
    rows=[];history=[];prediction={}
    for seed in SEEDS:
        for width in WIDTHS:
            torch.manual_seed(seed)
            model=torch.nn.Sequential(torch.nn.Linear(4,width),torch.nn.Tanh(),torch.nn.Linear(width,1))
            optimizer=torch.optim.Adam(model.parameters(),lr=.01)
            start=time.perf_counter()
            for step in range(STEPS):
                optimizer.zero_grad();loss=(model(tx)-ty).square().mean();loss.backward();optimizer.step()
                if step%50==0 or step==STEPS-1:history.append(dict(seed=seed,width=width,step=step+1,train_mse=float(loss.detach())))
            seconds=time.perf_counter()-start
            with torch.no_grad():
                pred=model(xx).numpy();test=float(((pred-data['test_y'])**2).mean());validation=float((model(vx)-vy).square().mean());train=float((model(tx)-ty).square().mean())
            p=sum(a.numel() for a in model.parameters());prediction[f'w{width}_s{seed}']=pred
            rows.append(dict(seed=seed,width=width,parameters=p,unique_training_examples=len(tx),sample_presentations=len(tx)*STEPS,steps=STEPS,train_mse=train,validation_mse=validation,test_mse=test,training_seconds=seconds,held_out_scale=width==64))
    np.savez(OUT/'predictions.npz',**prediction)
    mean=np.array([np.mean([r['test_mse'] for r in rows if r['width']==w]) for w in WIDTHS]);params=np.array([6*w+1 for w in WIDTHS])
    a,b=fit_curve(params[:-1],mean[:-1]);pred=np.exp(a+b*np.log(params));res=np.log(mean[:-1])-np.log(pred[:-1])
    rng=np.random.default_rng(909);curves=[];slopes=[]
    # Paired bootstrap: resample seed identities, keeping each seed's complete width series.
    matrix=np.array([[next(r['test_mse'] for r in rows if r['seed']==s and r['width']==w) for w in WIDTHS] for s in SEEDS])
    for _ in range(1000):
        sample=matrix[rng.integers(0,len(SEEDS),len(SEEDS))].mean(axis=0);aa,bb=fit_curve(params[:-1],sample[:-1]);curves.append(np.exp(aa+bb*np.log(params)));slopes.append(bb)
    band=np.quantile(curves,[.025,.975],axis=0)
    metrics=dict(protocol='Fixed data, optimizer, steps; widths 4/8/16/32 fit, width 64 held out. Test set used to characterize scale, not to select settings. Not a compute-optimal experiment.',widths=WIDTHS,seeds=SEEDS,parameters=params.tolist(),mean_test_mse=mean.tolist(),log_fit_intercept=a,log_fit_slope=b,in_sample_log_residuals=res.tolist(),predicted_mse=pred.tolist(),seed_bootstrap_95_low=band[0].tolist(),seed_bootstrap_95_high=band[1].tolist(),seed_bootstrap_slope_95=np.quantile(slopes,[.025,.975]).tolist(),heldout_relative_error=float((pred[-1]-mean[-1])/mean[-1]),noise_variance=.01,records=rows)
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
    for filename,records in [('runs.csv',rows),('training_history.csv',history)]:
        with (OUT/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(params,mean,'o-',label='Measured: seed mean');ax.plot(params,pred,'--',label='Local log-log fit');ax.fill_between(params,band[0],band[1],alpha=.2,label='Seed bootstrap interval');ax.axvline(params[-2],color='gray',linestyle=':');ax.set(xscale='log',yscale='log',xlabel='Parameter count',ylabel='Test MSE',title='Fixed-data width study: final scale is held out');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(FIG/'01_scaling.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(8,3.4));ax[0].axhline(0,color='gray');ax[0].plot(params[:-1],res,'o-');ax[0].set(xlabel='Parameters (fit scales)',ylabel='Log residual');
    for w in WIDTHS:
        rs=[r for r in rows if r['width']==w];ax[1].scatter([r['training_seconds'] for r in rs],[r['test_mse'] for r in rs],label=str(w))
    ax[1].set(xlabel='Measured CPU training seconds',ylabel='Test MSE');ax[1].legend(title='Width',fontsize=7);fig.tight_layout();fig.savefig(FIG/'02_diagnostics.png',dpi=160);plt.close(fig)
    env=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device='CPU',threads=1,platform=platform.system(),machine=platform.machine(),timing_scope='training loop only; excludes initialization, evaluation, plotting and notebook startup')
    (ROOT/'environment.json').write_text(json.dumps(env,indent=2));print(json.dumps({k:v for k,v in metrics.items() if k!='records'},indent=2))
if __name__=='__main__':main()
