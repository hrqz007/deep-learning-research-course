"""T4: offline small-network ensembles and conditional last-layer Bayes.
Every numeric dataset is generated locally; no network or framework is used.
"""
from pathlib import Path  # Construct portable local file paths.
import argparse  # Parse the learner's output directory.
import json  # Save human-readable metrics.
import numpy as np  # Operate on explicit numerical arrays.

NOISE = 0.18  # Known observation standard deviation in this synthetic world.
SEEDS = (11, 23, 37, 53, 71)  # Fixed independent initialization seeds.


def truth(x):
    """Known smooth data mechanism; not available to the fitting function."""
    return np.sin(1.6 * x) + 0.3 * x  # Curvature plus a trend beyond training support.


def make_data(seed=7601):
    rng = np.random.default_rng(seed)  # Isolate randomness from global state.
    x = rng.uniform(-2.0, 2.0, 100)  # Training support is only [-2, 2].
    y = truth(x) + rng.normal(0, NOISE, len(x))  # Independent observation noise.
    grid = np.linspace(-6.0, 6.0, 601)  # Evaluate the full declared range once.
    test_y = truth(grid) + rng.normal(0, NOISE, len(grid))  # Fresh observations.
    return x, y, grid, test_y  # No evaluation label is supplied to fitting.


def init(seed, width=16):
    rng = np.random.default_rng(seed)  # Each ensemble member gets one seed.
    return [rng.normal(0, 0.7, width), rng.normal(0, 0.2, width),
            rng.normal(0, 0.2, width), np.zeros(1)]  # w, b, v, output offset.


def forward(x, params):
    w, b, v, c = params  # Hidden slopes, offsets, output weights, output offset.
    h = np.tanh(np.asarray(x)[:, None] * w + b)  # Shape: samples by 16 units.
    return h @ v + c[0]  # Weighted hidden outputs yield one mean per sample.


def loss_grad(x, y, params, decay=1e-4):
    w, b, v, c = params  # Explicit arrays make the chain rule inspectable.
    h = np.tanh(x[:, None] * w + b)  # Store hidden values for differentiation.
    residual = h @ v + c[0] - y  # Prediction minus observed target.
    delta = 2.0 * residual / len(x)  # Derivative of mean squared error.
    hidden = delta[:, None] * v * (1.0 - h * h)  # tanh derivative via chain rule.
    grads = [(hidden * x[:, None]).sum(0) + 2 * decay * w,
             hidden.sum(0) + 2 * decay * b,
             h.T @ delta + 2 * decay * v, np.array([delta.sum()])]
    loss = np.mean(residual**2) + decay * sum(np.sum(p*p) for p in params[:3])
    return float(loss), grads  # Output offset is deliberately unregularized.


def train(x, y, seed, steps=2200):
    params = init(seed)  # A real trainable nonlinear network, not fake curves.
    first = [np.zeros_like(p) for p in params]  # Adam first moments.
    second = [np.zeros_like(p) for p in params]  # Adam squared moments.
    trace = []  # Keep actual losses, including optimization failures if any.
    for step in range(1, steps + 1):
        loss, grads = loss_grad(x, y, params)  # Differentiate all network weights.
        for k, grad in enumerate(grads):
            first[k] = 0.9 * first[k] + 0.1 * grad  # Smooth the gradient.
            second[k] = 0.999 * second[k] + 0.001 * grad**2  # Smooth its square.
            mh = first[k] / (1 - 0.9**step)  # Remove initialization bias.
            vh = second[k] / (1 - 0.999**step)  # Remove squared-moment bias.
            params[k] -= 0.015 * mh / (np.sqrt(vh) + 1e-8)  # Adam update.
        if step == 1 or step % 100 == 0:
            trace.append([step, loss])  # Save progress without selecting a winner.
    return params, trace  # Every predeclared member is kept.


def features(x, params):
    w, b, _, _ = params  # Freeze the learned hidden representation.
    h = np.tanh(np.asarray(x)[:, None] * w + b)  # Nonlinear features.
    return np.column_stack([h, np.ones(len(h))])  # Include an intercept feature.


def last_layer(x, y, grid, params, precision=1.0):
    if precision <= 0:  # Real checks remain active under python -O.
        raise ValueError('prior precision must be positive')
    phi = features(x, params)  # Training design matrix: 100 by 17.
    pg = features(grid, params)  # Evaluation design matrix: 601 by 17.
    hessian = precision * np.eye(phi.shape[1]) + phi.T @ phi / NOISE**2
    mean = np.linalg.solve(hessian, phi.T @ y / NOISE**2)  # Posterior/MAP mean.
    covariance = np.linalg.solve(hessian, np.eye(len(mean)))  # Conditional covariance.
    covariance = (covariance + covariance.T) / 2  # Remove roundoff asymmetry.
    pred = pg @ mean  # Posterior mean of the latent function.
    epistemic = np.einsum('ij,jk,ik->i', pg, covariance, pg)  # phi C phi^T.
    return pred, np.maximum(epistemic, 0), mean, covariance  # No added noise yet.


def score(y, mean, variance):
    if np.any(variance <= 0):  # Prevent log of zero and imaginary widths.
        raise ValueError('variance must be positive')
    error = y - mean  # All scores compare with noisy observations, not truth.
    return {'rmse': float(np.sqrt(np.mean(error**2))),
            'gaussian_nll': float(np.mean(0.5*np.log(2*np.pi*variance) + error**2/(2*variance))),
            'coverage_95': float(np.mean(np.abs(error) <= 1.96*np.sqrt(variance))),
            'mean_width_95': float(np.mean(3.92*np.sqrt(variance)))}


def run(output='outputs'):
    out = Path(output)  # User-controlled relative or absolute destination.
    out.mkdir(parents=True, exist_ok=True)  # Only experiment outputs are written.
    x, y, grid, test_y = make_data()  # Produce all documented local data.
    models, traces = zip(*(train(x, y, seed) for seed in SEEDS))  # Five full fits.
    member = np.stack([forward(grid, p) for p in models])  # Shape: 5 by 601.
    bayes, epi, mean, cov = last_layer(x, y, grid, models[0])  # Seed 11 features.
    means = np.stack([member[0], member.mean(0), bayes])  # Same evaluation points.
    variances = np.stack([np.full(len(grid), NOISE**2),
                          NOISE**2 + member.var(0), NOISE**2 + epi])  # Population mixture variance.
    names = ['single', 'ensemble', 'last_layer_bayes']  # Explicit modeling differences.
    regions = {'in_domain': abs(grid) <= 2, 'out_of_domain': abs(grid) > 2,
               'far_out': abs(grid) >= 4}  # Declared before examining results.
    metrics = {name: {region: score(test_y[mask], means[k, mask], variances[k, mask])
                     for region, mask in regions.items()} for k, name in enumerate(names)}
    rng = np.random.default_rng(7602)  # Independent Monte Carlo integration seed.
    draws = rng.multivariate_normal(mean, cov, size=4000)  # Conditional posterior draws.
    mc_mean = features(grid, models[0]) @ draws.mean(0)  # Monte Carlo latent mean.
    metrics['diagnostics'] = {'mc_draws': 4000,
        'mc_mean_max_abs_error': float(np.max(abs(mc_mean-bayes))),
        'bayes_scope': 'conditional on fitted features; not full-network posterior',
        'ensemble_interval': 'moment-matched Gaussian; not exact mixture quantiles',
        'steps_per_member': 2200, 'members': 5, 'data_seed': 7601}
    weights = {f'member_{j}_{k}': a for j, p in enumerate(models) for k, a in enumerate(p)}
    np.savez_compressed(out/'weights.npz', **weights, posterior_mean=mean, posterior_covariance=cov)
    np.savez_compressed(out/'data.npz', train_x=x, train_y=y, grid_x=grid, test_y=test_y, truth=truth(grid))
    np.savez_compressed(out/'predictions.npz', means=means, variances=variances,
                        members=member, epistemic=epi, mc_mean=mc_mean, traces=np.array(traces))
    loaded = np.load(out/'weights.npz')  # Roundtrip check reads actual saved arrays.
    reconstructed = [loaded[f'member_0_{k}'] for k in range(4)]
    metrics['diagnostics']['weight_reload_max_abs_error'] = float(np.max(abs(forward(grid,reconstructed)-member[0])))
    (out/'results.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics  # Notebook and CLI share the same implementation.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)  # Show local usage.
    parser.add_argument('--output', default='outputs')  # Avoid modifying frozen outputs when exploring.
    args = parser.parse_args()  # Accept one named destination argument.
    print(json.dumps(run(args.output), indent=2))  # Report actual measured results.
