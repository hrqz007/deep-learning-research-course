"""Replay every prediction model and all 84 actual control trajectories."""
from pathlib import Path
import argparse,json
import numpy as np
import experiment as e

def main(output='outputs'):
    out=Path(output);data=np.load(out/'data.npz');weights=np.load(out/'weights.npz');pred=np.load(out/'predictions.npz');met=json.loads((out/'metrics.json').read_text())
    prediction_checks=0;max_prediction_error=0.;max_dynamics_residual=0.
    for r in met['runs']:
        key=f"{r['method']}_{r['seed']}";w=weights[key]
        for domain,stem in [('validation','validation'),('in_domain','test'),('extrapolation','extra')]:
            st=data[stem+'_states'];ac=data[stem+'_actions'];pr=e.rollout(st[:,0],ac[:,:20],w)
            mse=np.mean((pr-st[:,1:21])**2,axis=(0,2));one=np.mean((e.step_model(st[:,:-1],ac,w)-st[:,1:])**2)
            np.testing.assert_allclose(mse,r[domain]['mse_by_horizon'],atol=1e-12);np.testing.assert_allclose(one,r[domain]['one_step_mse'],atol=1e-12)
            if domain!='validation':
                expected=pred[f'{domain}_{key}'];np.testing.assert_allclose(pr,expected,atol=1e-12);max_prediction_error=max(max_prediction_error,float(np.max(np.abs(pr-expected))))
            prediction_checks+=1
    for row in met['control_episodes']:
        method,i=row['method'],row['episode'];st=pred[f'episode_{method}_{i}_states'];ac=pred[f'episode_{method}_{i}_actions']
        np.testing.assert_allclose(st[0],[row['initial_x'],row['initial_v']],atol=1e-12)
        actual_next=e.true_step(st[:-1],ac);res=float(np.max(np.abs(actual_next-st[1:])));max_dynamics_residual=max(max_dynamics_residual,res)
        np.testing.assert_allclose(actual_next,st[1:],atol=1e-12)
        cost=np.sum(st[1:,0]**2+.1*st[1:,1]**2+.03*ac**2)
        failure=bool(np.any(np.abs(st[-10:,0])>.15) or np.any(np.abs(st[-10:,1])>.15))
        np.testing.assert_allclose(cost,row['cost'],atol=1e-12)
        if failure!=row['failure']:raise RuntimeError('Failure metric mismatch')
        w=weights['multi_step_11'] if method in ['multi_step','partial'] else weights['one_step_11']
        rerun=e.closed_loop(st[0],method,w)
        np.testing.assert_allclose(rerun['actions'],ac,atol=1e-12)
        np.testing.assert_allclose(rerun['states'],st,atol=1e-12)
    report={'status':'passed','model_domain_predictions_replayed':prediction_checks,'control_episodes_replayed':len(met['control_episodes']),
            'max_prediction_error':max_prediction_error,'max_true_dynamics_residual':max_dynamics_residual,'cost_failure_action_replay':'passed'}
    (out/'replay-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs');main(p.parse_args().output)
