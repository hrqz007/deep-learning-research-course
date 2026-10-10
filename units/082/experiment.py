"""Reproducible inverse problem: saturation, profile least squares and noise.
Only NumPy; the NIST observed data are kept separate from synthetic truth.
"""
from pathlib import Path
import numpy as np
import json,time,hashlib
ROOT=Path(__file__).resolve().parent

def forward(x,a,b):return a*(-np.expm1(-b*np.asarray(x)))
def fit(x,y,lam=0.,a0=2.4):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if x.ndim!=1 or y.shape!=x.shape or len(x)<2:
        raise ValueError("matching one-dimensional x,y with at least two observations required")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x<0) or np.ptp(x)==0:
        raise ValueError("finite nonnegative, varying pressures and finite responses required")
    if not np.isfinite(lam) or lam<0 or not np.isfinite(a0):
        raise ValueError("finite nonnegative regularization and finite prior required")
    # For fixed b, a has an analytic least-squares minimizer. Search log(b).
    def objective(logb):
        b=np.exp(logb);g=-np.expm1(-b*x)
        a=(g@y+lam*a0)/(g@g+lam)
        return float(np.sum((a*g-y)**2)+lam*(a-a0)**2),a,b
    # Coarse global scan then golden-section refinement avoids blind local starts.
    logs=np.linspace(np.log(1e-4),np.log(30.),180)
    values=np.array([objective(t)[0] for t in logs]); k=values.argmin()
    lo=logs[max(k-1,0)];hi=logs[min(k+1,len(logs)-1)];ratio=(np.sqrt(5)-1)/2
    c=hi-ratio*(hi-lo);d=lo+ratio*(hi-lo)
    for _ in range(65):
        if objective(c)[0]<objective(d)[0]:hi=d;d=c;c=hi-ratio*(hi-lo)
        else:lo=c;c=d;d=lo+ratio*(hi-lo)
    loss,a,b=objective((lo+hi)/2)
    return np.array([a,b]),loss

def sensitivity(x,a,b):
    # Derivatives w.r.t. log parameters are dimensionless relative changes.
    return np.column_stack([forward(x,a,b),a*b*x*np.exp(-b*x)])
def read_real():
    raw=(ROOT/'data/Misra1a.dat').read_text()
    part=raw.split('Data:   y               x')[1]
    data=np.array([[float(v) for v in line.split()] for line in part.splitlines() if line.strip()])
    if data.shape!=(14,2):raise ValueError("Unexpected NIST data shape")
    return data[:,1],data[:,0]

def run():
    start=time.perf_counter();rng=np.random.default_rng(82)
    (ROOT/'outputs').mkdir(exist_ok=True);(ROOT/'data').mkdir(exist_ok=True)
    results={'seed':82,'synthetic_truth':[2.4,.55],'noise_sigma':.01,'replicates':200,'synthetic':{}}
    allfits={}
    for label,xmax in [('narrow',.05),('medium',.8),('wide',5.)]:
        x=np.linspace(.005,xmax,24);truth=forward(x,2.4,.55)
        fits=[];regularized=[];wrong_prior=[]
        for _ in range(200):
            y=truth+rng.normal(0,.01,len(x));fits.append(fit(x,y)[0])
            if label=='narrow':
                regularized.append(fit(x,y,lam=.002,a0=2.4)[0]);wrong_prior.append(fit(x,y,lam=.002,a0=1.2)[0])
        fits=np.array(fits);allfits[label]=fits
        sv=np.linalg.svd(sensitivity(x,2.4,.55),compute_uv=False)
        def describe(arr):
            arr=np.asarray(arr)
            return {'median':np.median(arr,axis=0).tolist(),'q025':np.quantile(arr,.025,axis=0).tolist(),'q975':np.quantile(arr,.975,axis=0).tolist(),'relative_parameter_rmse':np.sqrt(np.mean(((arr-[2.4,.55])/[2.4,.55])**2,axis=0)).tolist()}
        item=describe(fits);item['sensitivity_condition']=float(sv[0]/sv[-1]);item['range']=[float(x[0]),xmax]
        item['b_search_boundary_fraction']=float(np.mean((fits[:,1]<1.01e-4)|(fits[:,1]>29.99)))
        if label=='narrow':item['regularized_correct_prior']=describe(regularized);item['regularized_wrong_prior']=describe(wrong_prior)
        results['synthetic'][label]=item
        np.savetxt(ROOT/'data'/f'synthetic_{label}.csv',np.column_stack([x,truth,y]),delimiter=',',header='x_dimensionless,truth,last_noisy_replicate',comments='')
        np.savetxt(ROOT/'outputs'/f'fits_{label}.csv',fits,delimiter=',',header='a,b',comments='')
    # Exact nonidentifiability when unknown sensor gain multiplies amplitude.
    gx=np.linspace(0,5,101)
    results['gain_equivalence_max_error']=float(np.max(abs(1.0*forward(gx,2.4,.55)-2.0*forward(gx,1.2,.55))))
    p,v=read_real(); x=p/1000.;y=v/100.;theta,sse=fit(x,y)
    train=np.arange(10);test=np.arange(10,14);partial,_=fit(x[train],y[train])
    linear=np.linalg.lstsq(np.column_stack([np.ones(10),x[train]]),y[train],rcond=None)[0]
    rmse=lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
    sigma=np.sqrt(sse/(len(x)-2));bootstrap=np.array([fit(x,forward(x,*theta)+rng.normal(0,sigma,len(x)))[0] for _ in range(300)])
    # Convert parameters and errors back to the recorded source units.
    original=theta*np.array([100.,.001]);ci=np.quantile(bootstrap,[.025,.975],axis=0)*[100.,.001]
    results['real']={'dataset':'NIST StRD Misra1a','kind':'observed','n':14,'full_fit_parameters_original_units':original.tolist(),
        'sse_original_units_squared':float(sse*10000),'residual_sigma_original_units':float(sigma*100),
        'conditional_parametric_bootstrap_95pct':ci.tolist(),'bootstrap_replicates':300,
        'heldout_high_pressure_n':4,'heldout_saturation_rmse':100*rmse(forward(x[test],*partial),y[test]),
        'heldout_linear_rmse':100*rmse(linear[0]+linear[1]*x[test],y[test]),
        'boundary_zero_error':float(abs(forward(0.,*theta))),
        'residual_sign_runs':int(1+np.count_nonzero(np.diff(np.sign(y-forward(x,*theta))))),
        'conservation':'not applicable: static adsorption response, no mass-flow/time information',
        'extrapolated_pressure_2000_prediction':float(forward(2.,*theta)*100),
        'sensitivity_condition':float(np.linalg.cond(sensitivity(x,*theta)))}
    np.savetxt(ROOT/'outputs/real_predictions.csv',np.column_stack([p,v,forward(x,*theta)*100,forward(x,*partial)*100,(linear[0]+linear[1]*x)*100]),delimiter=',',header='pressure_source_units,volume_source_units,full_fit,low_pressure_fit,linear_low_pressure_fit',comments='')
    np.savetxt(ROOT/'outputs/real_bootstrap.csv',bootstrap*[100.,.001],delimiter=',',header='amplitude_source_volume_units,rate_inverse_source_pressure_units',comments='')
    results['elapsed_seconds']=time.perf_counter()-start
    (ROOT/'outputs/metrics.json').write_text(json.dumps(results,indent=2));return results
if __name__=='__main__':print(json.dumps(run(),indent=2))
