"""DL091: learned dynamics, rollouts and receding-horizon control, pure NumPy."""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/dl091-matplotlib')
import argparse,csv,json,platform,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
DT=.15
SEEDS=[11,23,37]


def true_step(state, action):
    state=np.asarray(state); action=np.asarray(action)
    x,v=state[...,0],state[...,1]
    return np.stack([x+DT*v,.95*v+DT*(action-.25*x-.15*x**3)],axis=-1)


def generate(seed, trajectories=48, steps=40, extrapolate=False):
    rng=np.random.default_rng(seed)
    states=np.zeros((trajectories,steps+1,2));actions=rng.uniform(-1,1,(trajectories,steps))
    states[:,0,0]=rng.uniform(-1,1,trajectories)
    states[:,0,1]=rng.uniform(-.4,.4,trajectories)
    if extrapolate:
        states[:,0,0]=rng.choice([-1,1],trajectories)*rng.uniform(1.8,2.8,trajectories)
        actions*=1.8
    for t in range(steps):states[:,t+1]=true_step(states[:,t],actions[:,t])
    return states,actions


def design(state,action):
    return np.concatenate([state,np.asarray(action)[...,None],np.ones((*state.shape[:-1],1))],axis=-1)


def fit_one_step(states,actions,ridge=1e-6):
    z=design(states[:,:-1],actions).reshape(-1,4);target=states[:,1:].reshape(-1,2)
    return np.linalg.solve(z.T@z+ridge*np.eye(4),z.T@target)


def step_model(state,action,w):return design(state,action)@w


def windows(states,actions,horizon=5):
    st=[];ac=[];tar=[]
    for t in range(actions.shape[1]-horizon+1):
        st.append(states[:,t]);ac.append(actions[:,t:t+horizon]);tar.append(states[:,t+1:t+horizon+1])
    return np.concatenate(st),np.concatenate(ac),np.concatenate(tar)


def rollout(state,actions,w=None):
    pred=[];s=state.copy()
    for t in range(actions.shape[1]):
        s=true_step(s,actions[:,t]) if w is None else step_model(s,actions[:,t],w)
        pred.append(s)
    return np.stack(pred,axis=1)


def multistep_loss_grad(w,s0,actions,targets):
    """Full BPTT through an affine recurrent transition; no teacher forcing."""
    s=s0;zs=[];pred=[];n,h,_=targets.shape
    for t in range(h):
        z=design(s,actions[:,t]);zs.append(z);s=z@w;pred.append(s)
    pred=np.stack(pred,axis=1);err=pred-targets;loss=np.mean(err**2)
    grad=np.zeros_like(w);carry=np.zeros((n,2))
    for t in range(h-1,-1,-1):
        adj=2*err[:,t]/err.size+carry
        grad+=zs[t].T@adj
        carry=adj@w[:2].T
    return float(loss),grad


def fit_multistep(w,states,actions,steps=350):
    s0,a,target=windows(states,actions,5);w=w.copy();m=np.zeros_like(w);v=np.zeros_like(w);losses=[]
    for step in range(1,steps+1):
        loss,g=multistep_loss_grad(w,s0,a,target)
        norm=np.linalg.norm(g);g=g*min(1,10/(norm+1e-12))
        m=.9*m+.1*g;v=.999*v+.001*g*g
        w-=.003*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        if step%10==0 or step==1:losses.append(loss)
    return w,losses


def candidate_actions(horizon):
    grid=np.linspace(-1,1,9)
    if horizon==1:return grid[:,None]
    seq=[]
    for a in grid:
        for b in grid:seq.append([a]*(horizon//2)+[b]*(horizon-horizon//2))
    return np.array(seq)


def plan(state,w=None,horizon=16):
    actions=candidate_actions(horizon);s=np.repeat(np.asarray(state)[None],len(actions),axis=0);cost=np.zeros(len(actions))
    for t in range(horizon):
        s=true_step(s,actions[:,t]) if w is None else step_model(s,actions[:,t],w)
        cost+=s[:,0]**2+.1*s[:,1]**2+.03*actions[:,t]**2
    cost+=8*s[:,0]**2+.5*s[:,1]**2
    return float(actions[np.argmin(cost),0])


def closed_loop(initial,controller,w=None,steps=55):
    s=np.asarray(initial,dtype=float);states=[s.copy()];actions=[]
    for t in range(steps):
        if controller=='zero':a=0.
        elif controller=='pd':a=float(np.clip(-1.4*s[0]-1.1*s[1],-1,1))
        elif controller=='oracle_short':a=plan(s,None,1)
        elif controller=='partial':a=plan(np.array([s[0],0.]),w,16)
        else:a=plan(s,None if controller=='oracle' else w,16)
        s=true_step(s,a);states.append(s.copy());actions.append(a)
    states=np.array(states);actions=np.array(actions)
    cost=float(np.sum(states[1:,0]**2+.1*states[1:,1]**2+.03*actions**2))
    last=states[-10:]
    failure=bool(np.any(np.abs(last[:,0])>.15) or np.any(np.abs(last[:,1])>.15))
    return {'states':states,'actions':actions,'cost':cost,'failure':failure,
            'max_abs_x':float(np.abs(states[:,0]).max()),'terminal_distance':float(np.linalg.norm(states[-1]))}


def partial_observability(states,actions):
    # Compare identical x,a information budget with and without a two-position history.
    x=states[:,:-1,0];nxt=states[:,1:,0]
    z0=np.stack([x,actions,np.ones_like(x)],axis=-1)
    z1=np.stack([states[:,1:-1,0],states[:,:-2,0],actions[:,:-1],np.ones_like(actions[:,1:])],axis=-1)
    return z0.reshape(-1,3),nxt.reshape(-1),z1.reshape(-1,4),states[:,2:,0].reshape(-1)


def make_figures(out):
    out=Path(out);figs=out.parent/'figures';figs.mkdir(exist_ok=True)
    data=np.load(out/'data.npz');p=np.load(out/'predictions.npz');met=json.loads((out/'metrics.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for j,name in enumerate(['in_domain','extrapolation']):
        for method in ['one_step','multi_step']:
            arr=np.array([r[name]['mse_by_horizon'] for r in met['runs'] if r['method']==method])
            ax[j].plot(np.arange(1,21),arr.mean(0),label=method)
        ax[j].set(xlabel='Open-loop horizon',ylabel='State MSE',title=name.replace('_',' '));ax[j].legend()
    fig.savefig(figs/'01_rollout_errors.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for j,domain in enumerate(['in_domain','extrapolation']):
        true=p[domain+'_true'][0,:,0];ax[j].plot(np.arange(21),true,label='True',lw=3,c='black')
        for method in ['one_step','multi_step']:ax[j].plot(np.arange(1,21),p[f'{domain}_{method}_11'][0,:,0],label=method)
        ax[j].set(xlabel='Time step',ylabel='Position',title='Fixed-action rollout: '+domain);ax[j].legend(fontsize=8)
    fig.savefig(figs/'02_trajectory_predictions.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for method in ['zero','pd','one_step','multi_step','oracle','oracle_short','partial']:
        st=p['control_'+method+'_states'];ac=p['control_'+method+'_actions']
        ax[0].plot(np.arange(len(st))*DT,st[:,0],label=method)
        ax[1].plot(np.arange(len(ac))*DT,ac,label=method,lw=.9)
    ax[0].axhspan(-.15,.15,color='green',alpha=.1);ax[0].set(xlabel='Time (s)',ylabel='Position',title='Closed loop, same initial state');ax[0].legend(fontsize=7)
    ax[1].set(xlabel='Time (s)',ylabel='Applied action',title='Only first planned action is executed')
    fig.savefig(figs/'03_control_trajectories.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    controls=met['control_summary'];names=list(controls)
    ax[0].bar(names,[controls[n]['mean_cost'] for n in names],color='#16697a');ax[0].set(ylabel='Mean cumulative cost',title='Same objective and starts');ax[0].tick_params(axis='x',rotation=40)
    ax[1].bar(names,[controls[n]['failure_rate'] for n in names],color='#e29578');ax[1].set(ylabel='Failure rate',ylim=(0,1.05),title='Last 10 states must satisfy tolerances');ax[1].tick_params(axis='x',rotation=40)
    fig.savefig(figs/'04_cost_failure.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    tr=data['train_states'].reshape(-1,2);ex=data['extra_states'].reshape(-1,2)
    ax[0].scatter(ex[::5,0],ex[::5,1],s=8,alpha=.3,label='Extrapolation');ax[0].scatter(tr[::8,0],tr[::8,1],s=8,alpha=.3,label='Training')
    ax[0].set(xlabel='Position',ylabel='Velocity',title='Initial-state / action shift');ax[0].legend()
    d=met['partial_observation'];ax[1].bar(['x only','x history'],[d['position_only_mse'],d['history_mse']],color=['#e29578','#16697a'])
    ax[1].set(yscale='log',ylabel='One-step position MSE',title='History disambiguates hidden velocity')
    fig.savefig(figs/'05_partial_distribution.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3));ax.axis('off')
    boxes=[(.04,'Observed state\nor belief'),(.30,'Learned dynamics\nroll out candidates'),(.57,'Minimize predicted\ncost, choose first action'),(.83,'Real environment\nnew observation')]
    for x,label in boxes:ax.text(x,.55,label,ha='center',va='center',bbox=dict(boxstyle='round',facecolor='#d9edf0'),transform=ax.transAxes,fontsize=9)
    for x1,x2 in zip([.13,.41,.70],[.21,.48,.76]):ax.annotate('',xy=(x2,.55),xytext=(x1,.55),arrowprops=dict(arrowstyle='->'),xycoords='axes fraction')
    ax.annotate('Replan using actual feedback',xy=(.04,.3),xytext=(.83,.12),arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=180,angleB=-90,rad=15'),xycoords='axes fraction',ha='center')
    fig.savefig(figs/'06_planning_loop.png',dpi=160,bbox_inches='tight');plt.close(fig)


def main(output='outputs'):
    t0=time.perf_counter();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    train,a=generate(9101);val,av=generate(9102,16);test,at=generate(9103,32);extra,ae=generate(9104,32,extrapolate=True)
    np.savez_compressed(out/'data.npz',train_states=train,train_actions=a,validation_states=val,validation_actions=av,test_states=test,test_actions=at,extra_states=extra,extra_actions=ae)
    runs=[];weights={};preds={'in_domain_true':test[:,:21],'extrapolation_true':extra[:,:21]};control_rows=[]
    # Data bootstrap by whole trajectory quantifies training-data sensitivity, not optimizer randomness.
    for seed in SEEDS:
        idx=np.random.default_rng(seed).integers(0,len(train),len(train));s,acts=train[idx],a[idx]
        w1=fit_one_step(s,acts);wm,losses=fit_multistep(w1,s,acts)
        for method,w in [('one_step',w1),('multi_step',wm)]:
            weights[f'{method}_{seed}']=w
            ss,aa,tt=windows(s,acts,5)
            r={'seed':seed,'method':method,'training_five_step_mse':multistep_loss_grad(w,ss,aa,tt)[0],'multi_loss_history':losses if method=='multi_step' else []}
            for name,st,ac in [('validation',val,av),('in_domain',test,at),('extrapolation',extra,ae)]:
                pr=rollout(st[:,0],ac[:,:20],w);mse=np.mean((pr-st[:,1:21])**2,axis=(0,2))
                one=step_model(st[:,:-1],ac,w);r[name]={'one_step_mse':float(np.mean((one-st[:,1:])**2)),'mse_by_horizon':mse.tolist()}
                if name!='validation':preds[f'{name}_{method}_{seed}']=pr
            runs.append(r)
    z0,t0p,z1,t1p=partial_observability(train,a);b0=np.linalg.lstsq(z0,t0p,rcond=None)[0];b1=np.linalg.lstsq(z1,t1p,rcond=None)[0]
    zz0,tt0,zz1,tt1=partial_observability(test,at)
    partial={'position_only_mse':float(np.mean((zz0@b0-tt0)**2)),'history_mse':float(np.mean((zz1@b1-tt1)**2)),
             'same_x_opposite_velocity_next_x':true_step(np.array([[1.,.5],[1.,-.5]]),np.zeros(2))[:,0].tolist()}
    starts=np.array([[x,v] for x in [-2.,-1.,1.,2.] for v in [-.4,0.,.4]])
    for method in ['zero','pd','one_step','multi_step','oracle','oracle_short','partial']:
        for i,initial in enumerate(starts):
            w=weights['multi_step_11'] if method in ['multi_step','partial'] else weights['one_step_11']
            r=closed_loop(initial,method,w);control_rows.append({'method':method,'episode':i,'initial_x':float(initial[0]),'initial_v':float(initial[1]),**{k:r[k] for k in ['cost','failure','max_abs_x','terminal_distance']}})
            preds[f'episode_{method}_{i}_states']=r['states'];preds[f'episode_{method}_{i}_actions']=r['actions']
            if i==10:preds['control_'+method+'_states']=r['states'];preds['control_'+method+'_actions']=r['actions']
    control_summary={m:{'mean_cost':float(np.mean([r['cost'] for r in control_rows if r['method']==m])),
                         'failure_rate':float(np.mean([r['failure'] for r in control_rows if r['method']==m]))} for m in sorted(set(r['method'] for r in control_rows))}
    np.savez_compressed(out/'weights.npz',**weights,position_only=b0,position_history=b1)
    np.savez_compressed(out/'predictions.npz',**preds)
    with (out/'control.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(control_rows[0]));writer.writeheader();writer.writerows(control_rows)
    with (out/'prediction.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['method','seed','domain','one_step_mse','horizon_5_mse','horizon_20_mse'])
        for r in runs:
            for domain in ['validation','in_domain','extrapolation']:
                d=r[domain];writer.writerow([r['method'],r['seed'],domain,d['one_step_mse'],d['mse_by_horizon'][4],d['mse_by_horizon'][19]])
    result={'unit':'091','dt':DT,'seeds':SEEDS,'data_seeds':[9101,9102,9103,9104],
            'split_trajectories':[48,16,32,32],'multi_step_horizon':5,'multi_step_updates':350,'planner_horizon':16,'planner_candidates':81,
            'runs':runs,'partial_observation':partial,'control_summary':control_summary,'control_episodes':control_rows,
            'control_model_seed':11,'elapsed_seconds_before_plotting':time.perf_counter()-t0,
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},
            'limitations':['Affine model is deliberately misspecified','Oracle has true dynamics but limited action candidates','12 fixed starts, no population guarantee','Partial controller sets velocity to zero; not a learned belief filter','Closed-loop comparisons use model seed 11 only','Action/state shift is not every kind of OOD']}
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8');make_figures(out)
    print(json.dumps({'control':control_summary,'partial':partial,'seconds':result['elapsed_seconds_before_plotting']},indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs');main(p.parse_args().output)
