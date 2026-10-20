"""DL093: Gaussian denoising score matching, a linear baseline (not a neural net)."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/dl093-mpl')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MU = np.array([0.7, -0.4])
COV = np.array([[1.0, 0.55], [0.55, 1.8]])
SIGMA = 0.6

def score(x, sigma=SIGMA):
    """Row-vector convention: each row is one point; gradient has same shape."""
    return -(np.asarray(x) - MU) @ np.linalg.inv(COV + sigma**2 * np.eye(2))

def sample(rng, n, sigma=SIGMA):
    clean = rng.multivariate_normal(MU, COV, n)
    noise = rng.normal(size=(n, 2))
    noisy = clean + sigma * noise
    # Target is the conditional corruption score, not the marginal score.
    return clean, noisy, -noise / sigma

def fit(noisy, target):
    design = np.column_stack([noisy, np.ones(len(noisy))])
    return np.linalg.lstsq(design, target, rcond=None)[0]

def predict(x, weights):
    return np.column_stack([x, np.ones(len(x))]) @ weights

def main():
    out, figs = ROOT / 'outputs', ROOT / 'figures'
    out.mkdir(exist_ok=True); figs.mkdir(exist_ok=True)
    train_rng = np.random.default_rng(9301)
    test_rng = np.random.default_rng(9302)
    clean, train, targets = sample(train_rng, 60000)
    _, test, test_targets = sample(test_rng, 30000)
    weights = fit(train, targets)
    true = score(test); estimated = predict(test, weights)
    point_error = np.sum((estimated - true)**2, axis=1)
    precision = np.linalg.inv(COV + SIGMA**2 * np.eye(2))
    radius2 = np.einsum('ni,ij,nj->n', test-MU, precision, test-MU)
    groups = [radius2 < 2, (radius2 >= 2) & (radius2 < 6), radius2 >= 6]
    names = ['central_r2_lt_2', 'middle_r2_2_to_6', 'tail_r2_ge_6']
    metrics = {
        'seed_train': 9301, 'seed_test': 9302, 'n_train': len(train), 'n_test':len(test),
        'sigma': SIGMA, 'model':'affine least-squares DSM; not a deep network',
        'score_mse_per_coordinate': float(point_error.mean()/2),
        'test_dsm_mse_per_coordinate':float(np.mean((estimated-test_targets)**2)),
        'oracle_test_dsm_mse_per_coordinate':float(np.mean((true-test_targets)**2)),
        'coefficient_max_absolute_error':float(np.max(np.abs(weights[:2]+precision))),
        'groups': {name:{'n':int(mask.sum()),'score_mse_per_coordinate':float(point_error[mask].mean()/2)} for name,mask in zip(names,groups)}
    }
    # The grid is deliberately not sampled from p: this probes extrapolation.
    gx, gy = np.meshgrid(np.linspace(-4,5,41), np.linspace(-5,4,41))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    grid_score = score(grid); grid_pred = predict(grid,weights)
    r2 = np.einsum('ni,ij,nj->n',grid-MU,precision,grid-MU)
    density = np.exp(-r2/2)/(2*np.pi*np.sqrt(np.linalg.det(COV+SIGMA**2*np.eye(2))))
    metrics['uniform_grid_score_mse_per_coordinate'] = float(np.mean((grid_score-grid_pred)**2))
    np.savez_compressed(out/'data.npz', clean_train=clean, noisy_train=train,
        conditional_targets=targets, test=test, test_conditional_targets=test_targets,
        test_score=true, test_prediction=estimated, weights=weights, radius2=radius2,
        grid=grid, grid_score=grid_score, grid_prediction=grid_pred, grid_density=density)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(6,5)); ax.contourf(gx,gy,density.reshape(gx.shape),levels=20,cmap='Blues')
    k=np.arange(0,len(grid),3); ax.quiver(grid[k,0],grid[k,1],grid_score[k,0],grid_score[k,1],color='darkred',alpha=.7)
    ax.set(xlabel='x1',ylabel='x2',title='Noisy density and analytic score'); fig.tight_layout();fig.savefig(figs/'01_density_score.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4)); err=np.linalg.norm(grid_pred-grid_score,axis=1)
    m=ax[0].pcolormesh(gx,gy,err.reshape(gx.shape),shading='auto',cmap='magma');fig.colorbar(m,ax=ax[0],label='Vector error norm')
    ax[0].set(xlabel='x1',ylabel='x2',title='Grid score error')
    ax[1].bar(['Center','Middle','Tail'],[point_error[g].mean()/2 for g in groups]);ax[1].set(ylabel='MSE per coordinate',title='Independent held-out groups')
    fig.tight_layout();fig.savefig(figs/'02_error_regions.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4)); sigmas=[.2,.6,1.2]
    xs=np.linspace(-4,5,100); pts=np.column_stack([xs,np.full(len(xs),MU[1])])
    for s in sigmas: ax.plot(xs,score(pts,s)[:,0],label=f'sigma={s}')
    ax.axhline(0,color='gray',lw=.8);ax.set(xlabel='x1 (x2 fixed at mean)',ylabel='Score component 1',title='Smoothing changes the target score');ax.legend()
    fig.tight_layout();fig.savefig(figs/'03_noise_scales.png',dpi=150);plt.close(fig)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
    return metrics

if __name__ == '__main__':
    main()
