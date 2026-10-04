"""CPU float64 SVD tutorial. No network, fitting or GPU is used."""
from pathlib import Path
import argparse
import csv
import json
import math
import platform
import numpy as np

ROOT = Path(__file__).resolve().parent


def _reject_bool(value):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("Boolean is not a numerical measurement")
    if isinstance(value, (list, tuple)):
        for item in value:
            _reject_bool(item)
    elif isinstance(value, np.ndarray) and value.dtype.kind == 'b':
        raise TypeError("Boolean array is not accepted")


def matrix(value, name='matrix'):
    """Copy a nonempty 2D real numeric container into finite float64."""
    _reject_bool(value)
    raw = np.asarray(value)
    if raw.ndim != 2 or 0 in raw.shape:
        raise ValueError(f'{name} must be a nonempty 2D matrix')
    if raw.dtype.kind not in 'iuf':
        raise TypeError(f'{name} must contain real numeric values')
    with np.errstate(over='ignore', invalid='ignore'):
        a = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(a).all():
        raise ValueError(f'{name} must be finite after float64 conversion')
    return a


def finite(value, name):
    if not np.isfinite(value).all():
        raise ArithmeticError(f'{name} exceeds supported numerical range')
    return value


def reduced_svd(value):
    a = matrix(value)
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    finite(u, 'U'); finite(s, 'singular values'); finite(vt, 'V transpose')
    return u, s, vt


def rank_k(value, k):
    a = matrix(value)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError('k must be an integer, not Boolean')
    if not 0 <= k <= min(a.shape):
        raise ValueError('k must lie between zero and min(shape)')
    u, s, vt = reduced_svd(a)
    with np.errstate(over='ignore', invalid='ignore'):
        result = (u[:, :k] * s[:k]) @ vt[:k, :]
    return finite(result, 'reconstruction')


def frobenius(value):
    a = matrix(value)
    # math.hypot scales internally; squaring every entry first can overflow.
    result = math.hypot(*(float(v) for v in a.flat))
    return float(finite(result, 'Frobenius norm'))


def numerical_rank(value, tolerance):
    a = matrix(value)
    _reject_bool(tolerance)
    if not isinstance(tolerance, (float, int, np.floating, np.integer)):
        raise TypeError('tolerance must be a real scalar')
    tau = float(tolerance)
    if not math.isfinite(tau) or tau < 0:
        raise ValueError('tolerance must be finite and nonnegative')
    s = reduced_svd(a)[1]
    return int(np.count_nonzero(s > tau))


def condition2(value):
    """Square inverse-problem convention; no tolerance silently changes rank."""
    a = matrix(value)
    if a.shape[0] != a.shape[1]:
        raise ValueError('this lesson defines inverse condition only for square matrices')
    s = reduced_svd(a)[1]
    if s[-1] == 0:
        return math.inf
    with np.errstate(over='ignore', invalid='ignore'):
        result = s[0] / s[-1]
    return float(finite(result, 'condition ratio'))


def solve_row(value, right):
    """Solve x @ W = b by transposing; do not form an inverse."""
    w, b = matrix(value, 'W'), matrix(right, 'b')
    if w.shape[0] != w.shape[1] or b.shape != (1, w.shape[1]):
        raise ValueError('W must be square and b must have shape (1,n)')
    x = np.linalg.solve(w.T, b.T).T
    return finite(x, 'solution')


def perturbation_case(epsilon, delta):
    for v in (epsilon, delta):
        _reject_bool(v)
        if not isinstance(v, (int, float, np.integer, np.floating)):
            raise TypeError('epsilon and delta must be real scalars')
        if not math.isfinite(float(v)) or float(v) <= 0:
            raise ValueError('epsilon and delta must be finite positive scalars')
    w = np.diag([1.0, float(epsilon)])
    b = np.array([[1.0, 0.0]])
    db = np.array([[0.0, float(delta)]])
    x = solve_row(w, b)
    xp = solve_row(w, b + db)
    rel_b = frobenius(db) / frobenius(b)
    rel_x = frobenius(xp - x) / frobenius(x)
    ratio = finite(rel_x / rel_b, 'amplification')
    return {'epsilon':float(epsilon), 'delta':float(delta), 'condition2':condition2(w),
            'x':x.tolist(), 'perturbed_x':xp.tolist(), 'relative_b':rel_b,
            'relative_x':rel_x, 'amplification':float(ratio),
            'perturbed_residual':frobenius(xp @ w - (b + db))}


def run(output, config_path=None):
    cfg = json.loads((Path(config_path) if config_path else ROOT/'data/config.json').read_text())
    if not isinstance(cfg.get('epsilons'), list) or not cfg['epsilons']:
        raise ValueError('epsilons must be a nonempty list')
    w = matrix(cfg['matrix'])
    u, s, vt = reduced_svd(w)
    rows = []
    for k in range(min(w.shape) + 1):
        wk = rank_k(w, k)
        err = frobenius(w - wk)
        rows.append({'rank_budget':k, 'frobenius_error':err,
                     'tail_formula':math.hypot(*(float(t) for t in s[k:])),
                     'spectral_tail':float(s[k]) if k < len(s) else 0.0})
    perturb = [perturbation_case(e, cfg['delta']) for e in cfg['epsilons']]
    hidden = np.array([[10., 0.], [0., .1]])
    observations = np.array([[1., 1.], [1., -1.]])
    full = observations @ hidden
    compressed = observations @ rank_k(hidden, 1)
    result = {'singular_values':s.tolist(), 'svd_shapes':[list(z.shape) for z in (u,s,vt)],
              'reconstruction':rank_k(w, min(w.shape)).tolist(),
              'rank_one':rank_k(w, 1).tolist(), 'errors':rows,
              'perturbations':perturb,
              'task_counterexample':{'full_output':full.tolist(), 'rank_one_output':compressed.tolist(),
                                      'relative_frobenius_error':frobenius(hidden-rank_k(hidden,1))/frobenius(hidden)}}
    # Prepare the complete numerical report before creating output files.
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    (out/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    for filename, records in [('rank_errors.csv', rows), ('perturbations.csv', [{k:r[k] for k in ['epsilon','delta','condition2','relative_b','relative_x','amplification','perturbed_residual']} for r in perturb])]:
        with (out/filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    env = {'python':platform.python_version(), 'numpy':np.__version__, 'device':'CPU', 'dtype':'float64', 'network_used':False}
    (out/'environment.json').write_text(json.dumps(env, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'outputs')
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    result = run(args.output, args.config)
    print(json.dumps({'singular_values':result['singular_values'], 'rank_one':result['rank_one'], 'status':'passed'}, ensure_ascii=False))
