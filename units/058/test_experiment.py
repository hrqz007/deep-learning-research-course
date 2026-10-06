"""Exact numerics and full package checks, also enforced under Python -O."""
import json
from pathlib import Path
import numpy as np
import torch
from experiment import Autoencoder,make_data,reconstruction_loss,sparse_penalty,fit_probe,ROOT

def check(ok,msg):
    if not ok: raise AssertionError(msg)

def raises_value_error(fn,msg):
    try: fn()
    except ValueError: return
    raise AssertionError(msg)

def main(output=None):
    out=Path(output) if output is not None else ROOT/'outputs'
    torch.set_num_threads(1)
    x=torch.tensor([[2.,-1.]])
    e=torch.tensor([[.5],[-.5]],requires_grad=True)
    dec=torch.tensor([[1.,0.]],requires_grad=True)
    z=x@e;pred=z@dec;loss=reconstruction_loss(pred,x);loss.backward()
    check(np.isclose(loss.item(),.625),'hand MSE')
    check(np.allclose(dec.grad.numpy(),[[-.75,1.5]]),'decoder chain gradient')
    check(np.allclose(e.grad.numpy(),[[-1.],[.5]]),'encoder chain gradient')
    with torch.no_grad():
        updated=(x@(e-.1*e.grad))@(dec-.1*dec.grad)
    check(np.isclose(reconstruction_loss(updated,x).item(),.27900390625),'one SGD step')
    raises_value_error(lambda: reconstruction_loss(torch.ones(2,2),torch.ones(2,1)),'no broadcasting')
    raises_value_error(lambda: reconstruction_loss(torch.empty(0,2),torch.empty(0,2)),'no empty loss')
    check(np.isclose(sparse_penalty(torch.tensor([[2.,-1.]]),.1),.15),'L1 reduction')
    for coefficient in [-1.,float('nan'),float('inf')]:
        raises_value_error(lambda: sparse_penalty(torch.ones(1,1),coefficient),'invalid sparse penalty')
    matrix=np.array([[3,0],[-3,0],[0,1],[0,-1]],dtype=float)
    _,s,vt=np.linalg.svd(matrix,full_matrices=False)
    recon=matrix@vt[:1].T@vt[:1]
    check(np.isclose(np.mean((matrix-recon)**2),.25),'PCA discarded energy /ND')
    data=make_data()
    for split,n in [('train',640),('validation',160),('test',320)]:
        check(data['x_'+split].shape==(n,8),'data schema '+split)
        check(np.array_equal(np.unique(data['y_'+split],return_counts=True)[1],[n//2,n//2]),'balanced split '+split)
    check(sum(p.numel() for p in Autoencoder().parameters())==362,'parameter count')
    train=np.array([[0.,1.],[2.,1.],[4.,1.]])
    scores,params=fit_probe(train,np.array([-1.,1.,1.]),np.array([[100.,1.]]))
    check(np.allclose(params['mean'],[2.,1.]),'test must not enter normalization')
    check(params['scale'][1]==1.,'constant probe coordinate safe scale')
    raises_value_error(lambda: fit_probe(train,np.ones(3),train,ridge=0),'positive ridge')
    expected={f'weights_{name}_{seed}.npz' for name in ['ae','dae'] for seed in [5801,5802,5803]}
    found={p.name for p in out.glob('weights_*.npz')}
    check(found==expected,'all six exact checkpoints required')
    result=json.loads((out/'results.json').read_text())
    rows=result['results']
    expected_keys={('raw',0),('pca2',0)}|{(name,seed) for name in ['random2','ae','dae'] for seed in [5801,5802,5803]}
    check(len(rows)==11 and {(r['representation'],r['seed']) for r in rows}==expected_keys,'11 unique results')
    plots=np.load(out/'plot_data.npz',allow_pickle=False)
    stored=np.load(out/'data.npz',allow_pickle=False)
    for key,value in data.items(): check(np.array_equal(stored[key],value),'regenerated dataset '+key)
    mean=data['x_train'].mean(0);centered=data['x_test']-mean
    check(np.allclose(plots['target'],centered),'plot targets')
    for row in rows:
        key=f"{row['representation']}_{row['seed']}"
        check(row['probe_n']==320 and row['probe_correct']/320==row['probe_accuracy'],'accuracy denominator '+key)
        probe=np.load(out/f'probe_{key}.npz',allow_pickle=False)
        z=plots['latent_'+key]
        augmented=np.column_stack([(z.astype(float)-probe['mean'])/probe['scale'],np.ones(len(z))])
        scores=augmented@probe['weights']
        check(np.allclose(scores,probe['scores'],atol=1e-8),'saved probe scores '+key)
        check(int(np.sum(np.where(scores>=0,1.,-1.)==data['y_test']))==row['probe_correct'],'probe accuracy '+key)
        mse=float(np.mean((plots['recon_'+key]-centered)**2))
        check(np.isclose(mse,row['test_reconstruction_mse'],atol=1e-7),'recorded reconstruction '+key)
    for filename in sorted(expected):
        with np.load(out/filename,allow_pickle=False) as a:
            model=Autoencoder().eval()
            check(np.array_equal(a['input_mean'],mean),'checkpoint training mean')
            model.load_state_dict({k:torch.from_numpy(a[k].copy()) for k in a.files if k!='input_mean'})
            with torch.no_grad():
                inp=torch.from_numpy(centered)
                reconstructed=model(inp).numpy();latent=model.encoder(inp).numpy()
            key=filename[len('weights_'):-4]
            check(np.isfinite(reconstructed).all(),'checkpoint finite '+key)
            check(np.allclose(reconstructed,plots['recon_'+key],atol=2e-6),'checkpoint full test reconstruction '+key)
            check(np.allclose(latent,plots['latent_'+key],atol=2e-6),'checkpoint full test latent '+key)
            row=next(r for r in rows if f"{r['representation']}_{r['seed']}"==key)
            check([t['step'] for t in row['trace']]==[1,50,100,150,200,250],'complete trajectory '+key)
    pca=np.load(out/'pca.npz',allow_pickle=False)
    residual=np.sum(pca['singular_values'][2:]**2)/(640*8)
    check(np.isclose(residual,result['pca_train_optimum_mse'],atol=1e-7),'training PCA optimum')
    print('PASS: hand gradients and SGD, loss guards, L1, PCA, train-only probe, 11 rows, all six exact checkpoints and full-test replay')

if __name__=='__main__':main()
