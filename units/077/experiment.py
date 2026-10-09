"""T5: gradients, integrated gradients, sanity checks and known causal mechanisms."""
from pathlib import Path  # Local path handling only.
import argparse  # Read the output destination.
import json  # Serialize measured evidence.
import numpy as np  # Explicit small numerical arrays.


def generate(n=600, seed=7701):
    rng = np.random.default_rng(seed)  # Separate each dataset's randomness.
    u = rng.normal(size=n)  # Shared cause, called U in the causal diagram.
    ex = rng.normal(scale=0.4, size=n)  # Exogenous X noise.
    ey = rng.normal(scale=0.2, size=n)  # Exogenous Y noise.
    ez = rng.normal(scale=0.08, size=n)  # A precise proxy's noise.
    x = u + ex  # Structural equation for X.
    y = 2*x + 3*u + ey  # X has a genuine direct effect of 2 per unit.
    z = y + ez  # Z is a downstream measurement, not a cause of Y.
    return np.column_stack([x, z]), y, np.column_stack([u, ex, ey, ez])


def init(seed=17, width=12):
    rng = np.random.default_rng(seed)  # Reproducible weight initialization.
    return [rng.normal(0, 0.15, (2, width)), np.zeros(width),
            rng.normal(0, 0.2, width), np.zeros(1)]  # W, b, v, c.


def predict(x, p):
    w, b, v, c = p  # Label each array by its role.
    return np.tanh(np.asarray(x) @ w + b) @ v + c[0]  # Two input fields, one output.


def gradient(x, p):
    w, b, v, _ = p  # The output offset does not affect input sensitivity.
    h = np.tanh(np.asarray(x) @ w + b)  # Cache hidden activations.
    return ((1-h*h)*v) @ w.T  # Chain rule, shape: samples by two fields.


def ig(x, baseline, p, steps=512):
    if steps < 1:
        raise ValueError('integration steps must be positive')  # Active under -O.
    x = np.asarray(x, dtype=float)  # One point with exactly two input values.
    baseline = np.asarray(baseline, dtype=float)  # A declared comparison state.
    if x.shape != (2,) or baseline.shape != (2,):
        raise ValueError('x and baseline must each have two fields')
    alpha = (np.arange(steps)+0.5)/steps  # Midpoint quadrature avoids endpoints.
    path = baseline + alpha[:, None]*(x-baseline)  # Straight interpolation path.
    return (x-baseline)*gradient(path, p).mean(0)  # Path-average derivative times displacement.


def train(x, y, seed=17, steps=1800):
    p = init(seed)  # Train all parameters, including the representation.
    m = [np.zeros_like(a) for a in p]  # Adam gradient means.
    s = [np.zeros_like(a) for a in p]  # Adam squared-gradient means.
    trace = []  # Keep progress at fixed intervals.
    for t in range(1, steps+1):
        w, b, v, c = p  # Current network arrays.
        h = np.tanh(x@w+b)  # Forward hidden computation.
        err = h@v+c[0]-y  # Residual for every training row.
        d = 2*err/len(y)  # MSE derivative with respect to prediction.
        dh = d[:, None]*v*(1-h*h)  # Chain rule through the hidden units.
        grads = [x.T@dh, dh.sum(0), h.T@d, np.array([d.sum()])]
        for k,g in enumerate(grads):
            m[k] = .9*m[k]+.1*g  # Exponential gradient average.
            s[k] = .999*s[k]+.001*g*g  # Exponential squared average.
            p[k] -= .008*(m[k]/(1-.9**t))/(np.sqrt(s[k]/(1-.999**t))+1e-8)
        if t == 1 or t % 100 == 0:
            trace.append([t, float(np.mean(err*err))])  # Fixed logging rule.
    return p, trace


def slope(a, y, adjust=None):
    cols = [np.ones(len(y)), a]  # Intercept and the proposed causal field.
    if adjust is not None:
        cols.append(adjust)  # In this known world, U is the correct adjustment.
    return float(np.linalg.lstsq(np.column_stack(cols), y, rcond=None)[0][1])


def run(output='outputs'):
    out = Path(output)  # Default reproduces the supplied reference outputs.
    out.mkdir(parents=True, exist_ok=True)  # No network access is performed.
    x, y, exogenous = generate()  # Train only on X and the proxy Z.
    test_x, test_y, test_exo = generate(400, 7702)  # Separate test realization.
    p, trace = train(x, y)  # Train an actual nonlinear regression model.
    point = np.array([.8, 4.0])  # Predeclared pedagogical query, independent of test labels.
    base0 = np.zeros(2)  # Zero reference is a design choice, not neutral truth.
    base1 = np.array([.5, 2.5])  # Alternative plausible reference state.
    grad = gradient(point[None], p)[0]  # Local derivative at the query.
    attribution = ig(point, base0, p)  # Integrated gradients from zero.
    attribution1 = ig(point, base1, p)  # Same model and point, different baseline.
    random_all = init(77)  # Destroy every learned parameter.
    random_head = [a.copy() for a in p]  # Keep features for partial randomization.
    random_head[2:] = random_all[2:]  # Destroy just the trained output layer.
    probe = test_x[:80]  # Fixed unlabeled probe rows for aggregate checks.
    original_igs = np.stack([ig(row, base0, p, 128) for row in probe])
    checks = {}
    for name, rp in [('head', random_head), ('all', random_all)]:
        randomized = np.stack([ig(row, base0, rp, 128) for row in probe])
        denom = np.linalg.norm(original_igs)*np.linalg.norm(randomized)
        checks[name] = {'relative_l2_change': float(np.linalg.norm(original_igs-randomized)/np.linalg.norm(original_igs)),
                        'signed_cosine': float(np.sum(original_igs*randomized)/denom)}
    deltas = np.array([.001, .01, .1, 1.0])  # Probe where linearization stops working.
    changes = []
    for d in deltas:
        changed = point.copy()  # Hold the other model input fixed.
        changed[1] += d  # Perturb the proxy Z in model space.
        changes.append(float(predict(changed[None],p)[0]-predict(point[None],p)[0]))
    u, ex, ey, ez = test_exo.T  # Keep exogenous variables fixed for paired interventions.
    do_x0 = np.full(len(u), -0.5)  # Overwrite the structural equation for X.
    do_x1 = np.full(len(u), 0.5)  # A one-unit intervention contrast.
    y0 = 2*do_x0+3*u+ey  # Recompute the downstream outcome under do(X=-0.5).
    y1 = 2*do_x1+3*u+ey  # Recompute under do(X=0.5), same noise and U.
    z0, z1 = y0+ez, y1+ez  # The downstream proxy must also follow the outcome.
    do_z_shift = test_x.copy()  # In a Z intervention, X and Y remain unchanged.
    do_z_shift[:,1] += 1.0  # Replace only the proxy's structural equation.
    results = {'test_rmse': float(np.sqrt(np.mean((predict(test_x,p)-test_y)**2))),
        'point': point.tolist(), 'baseline_zero': base0.tolist(), 'baseline_alternative': base1.tolist(),
        'prediction': float(predict(point[None],p)[0]), 'gradient': grad.tolist(),
        'ig_zero': attribution.tolist(), 'ig_alternative': attribution1.tolist(),
        'completeness_error': float(abs(attribution.sum()-(predict(point[None],p)[0]-predict(base0[None],p)[0]))),
        'randomization': checks, 'perturbation': {'delta': deltas.tolist(), 'actual_change': changes,
                                                'linearized_change': (deltas*grad[1]).tolist()},
        'causal': {'observational_x_slope': slope(test_x[:,0],test_y),
                   'u_adjusted_x_slope': slope(test_x[:,0],test_y,u),
                   'do_x_effect_per_unit': float(np.mean(y1-y0)),
                   'do_z_effect_per_unit': 0.0,
                   'model_response_to_z_plus_one': float(np.mean(predict(do_z_shift,p)-predict(test_x,p))),
                   'model_response_to_coherent_x_intervention': float(np.mean(predict(np.column_stack([do_x1,z1]),p)-predict(np.column_stack([do_x0,z0]),p)))},
        'steps':1800, 'seeds': {'train_data':7701,'test_data':7702,'train':17,'randomization':77}}
    np.savez_compressed(out/'data.npz', train_x=x,train_y=y,train_exogenous=exogenous,
                        test_x=test_x,test_y=test_y,test_exogenous=test_exo,
                        do_x0=do_x0,do_x1=do_x1,y0=y0,y1=y1,z0=z0,z1=z1)
    np.savez_compressed(out/'weights.npz', **{f'p{k}':a for k,a in enumerate(p)})
    np.savez_compressed(out/'predictions.npz', test_prediction=predict(test_x,p),
                        original_igs=original_igs,trace=np.array(trace))
    saved=np.load(out/'weights.npz')  # Reload the actual serialized arrays.
    results['reload_max_abs_error']=float(np.max(abs(predict(test_x,[saved[f'p{k}'] for k in range(4)])-predict(test_x,p))))
    (out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    return results

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)  # Self-contained CLI help.
    parser.add_argument('--output',default='outputs')  # Separate exploratory output if desired.
    args=parser.parse_args()  # Parse without external configuration.
    print(json.dumps(run(args.output),indent=2))  # Print measured evidence only.
