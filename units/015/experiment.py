"""Unit 015: deterministic local models and quadratic gradient descent.

All parameter vectors use shape (1, 2). This is a small CPU mechanism lab,
not a neural-network benchmark. Fail explicitly on invalid numeric ranges.
"""
from pathlib import Path
import argparse
import csv
import io
import json
import platform
import numpy as np

ROOT = Path(__file__).resolve().parent


def _reject_boolean(value):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError('boolean is not a real teaching coordinate')
    if isinstance(value, np.ndarray):
        if value.dtype.kind == 'b':
            raise TypeError('boolean array is not a real coordinate')
        if value.dtype.kind == 'O':
            for leaf in value.flat:
                _reject_boolean(leaf)
    elif isinstance(value, (list, tuple)):
        for leaf in value:
            _reject_boolean(leaf)


def real_array(value, shape, name):
    _reject_boolean(value)
    original = np.asarray(value)
    if original.dtype.kind not in 'iuf':
        raise TypeError(name + ' must have a real numeric dtype')
    if original.shape != shape:
        raise ValueError(name + ' must have shape ' + str(shape))
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.asarray(original, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(name + ' must be finite after float64 conversion')
    return result.copy()


def row(value, name='point'):
    return real_array(value, (1, 2), name)


def symmetric(value):
    result = real_array(value, (2, 2), 'Hessian')
    if not np.array_equal(result, result.T):
        raise ValueError('Hessian must be exactly symmetric; no silent repair')
    return result


def scalar(value, name='scalar'):
    _reject_boolean(value)
    a = np.asarray(value)
    if a.shape != () or a.dtype.kind not in 'iuf':
        raise TypeError(name + ' must be a real scalar')
    with np.errstate(over='ignore', invalid='ignore'):
        v = float(a)
    if not np.isfinite(v):
        raise ValueError(name + ' must be finite')
    return v


def finite(value):
    if not np.all(np.isfinite(value)):
        raise ArithmeticError('result is outside finite float64 range')
    return value


def quadratic(point, hessian):
    p, h = row(point), symmetric(hessian)
    with np.errstate(over='ignore', invalid='ignore'):
        out = .5 * (p @ h @ p.T)[0, 0]
    return float(finite(out))


def gradient(point, hessian):
    p, h = row(point), symmetric(hessian)
    with np.errstate(over='ignore', invalid='ignore'):
        out = p @ h
    return finite(out)


def polynomial(point):
    """F(x,y)=x**4+y**2/2, smooth on R**2."""
    x, y = row(point)[0]
    with np.errstate(over='ignore', invalid='ignore'):
        out = x**4 + .5*y*y
    return float(finite(out))


def polynomial_gradient(point):
    x, y = row(point)[0]
    with np.errstate(over='ignore', invalid='ignore'):
        out = np.array([[4*x**3, y]])
    return finite(out)


def polynomial_hessian(point):
    x, _ = row(point)[0]
    with np.errstate(over='ignore', invalid='ignore'):
        out = np.diag([12*x*x, 1.0])
    return finite(out)


def steepening(point):
    """R(x,y)=x+x**2/2+10*x**4+y**2/2."""
    x, y = row(point)[0]
    with np.errstate(over='ignore', invalid='ignore'):
        out = x + .5*x*x + 10*x**4 + .5*y*y
    return float(finite(out))


def local_models(value, grad, hessian, delta):
    f, g, h, d = scalar(value), row(grad, 'gradient'), symmetric(hessian), row(delta, 'delta')
    with np.errstate(over='ignore', invalid='ignore'):
        first = f + (g @ d.T)[0, 0]
        second = first + .5*(d @ h @ d.T)[0, 0]
    return float(finite(first)), float(finite(second))


def step(point, hessian, eta):
    p, rate = row(point), scalar(eta, 'eta')
    g = gradient(p, hessian)
    with np.errstate(over='ignore', invalid='ignore'):
        out = p - rate*g
    return finite(out)


def spectrum(hessian):
    h = symmetric(hessian)
    values, vectors = np.linalg.eigh(h)
    finite(values); finite(vectors)
    return values, vectors


def spd_bounds(hessian):
    values, _ = spectrum(hessian)
    if values[0] <= 0:
        raise ValueError('strict positive definiteness is required')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        maximum_eta = 2.0/values[-1]
        condition = values[-1]/values[0]
    finite(maximum_eta); finite(condition)
    if maximum_eta <= 0:
        raise ArithmeticError('positive stability bound is not representable')
    return {'minimum_curvature': float(values[0]), 'maximum_curvature': float(values[-1]),
            'eta_strict_upper': float(maximum_eta), 'condition_number': float(condition)}


def count_steps(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError('iterations must be an integer')
    if not 0 <= value <= 5000:
        raise ValueError('iterations must be between 0 and 5000')
    return int(value)


def trajectory(initial, hessian, eta, iterations):
    p, h, rate = row(initial), symmetric(hessian), scalar(eta, 'eta')
    n = count_steps(iterations)
    records = []
    for k in range(n+1):
        records.append({'iteration': k, 'x': float(p[0, 0]), 'y': float(p[0, 1]),
                        'loss': quadratic(p, h)})
        if k < n:
            p = step(p, h, rate)
    return records


def finite_difference_hessian(gradient_function, point, step_size):
    """Differentiate each gradient component; rows index its components."""
    p, h = row(point), scalar(step_size, 'finite-difference step')
    if h <= 0:
        raise ValueError('finite-difference step must be positive')
    denominator = 2.0*h
    if not np.isfinite(denominator):
        raise ArithmeticError('finite-difference denominator overflow')
    result = np.empty((2, 2))
    for j in range(2):
        plus, minus = p.copy(), p.copy()
        with np.errstate(over='ignore', invalid='ignore'):
            plus[0, j] += h
            minus[0, j] -= h
        finite(plus); finite(minus)
        if plus[0, j] == p[0, j] or minus[0, j] == p[0, j]:
            raise ArithmeticError('finite-difference step does not change the input')
        gp = row(gradient_function(plus), 'computed gradient')
        gm = row(gradient_function(minus), 'computed gradient')
        with np.errstate(over='ignore', invalid='ignore'):
            result[:, j] = ((gp-gm)/denominator)[0]
    return finite(result)


def run(config_path=ROOT/'data/config.json', output=ROOT/'outputs'):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    h = symmetric(config['hessian']); p = row(config['initial_point'])
    bounds = spd_bounds(h)
    rates = [scalar(v, 'eta') for v in config['step_sizes']]
    if not rates or len(rates) > 20:
        raise ValueError('use between 1 and 20 step sizes')
    n = count_steps(config['iterations'])
    ts = [scalar(v, 'Taylor step') for v in config['taylor_steps']]
    if not ts or len(ts) > 50 or any(v <= 0 for v in ts):
        raise ValueError('use between 1 and 50 positive Taylor steps')
    records = []; summaries = []
    for rate in rates:
        series = trajectory(p, h, rate, n)
        records.extend(dict(eta=rate, **r) for r in series)
        values, _ = spectrum(h)
        with np.errstate(over='ignore', invalid='ignore'):
            factors = finite(1-rate*values).tolist()
        summaries.append({'eta': rate, 'factors': factors,
                          'initial_loss': series[0]['loss'], 'first_loss': series[min(1,n)]['loss'],
                          'final_loss': series[-1]['loss']})
    taylor = []; base = np.array([[1.0, 1.0]])
    for v in ts:
        d = np.array([[v, 2*v]])
        first, second = local_models(polynomial(base), polynomial_gradient(base), polynomial_hessian(base), d)
        actual = polynomial(base+d)
        taylor.append({'h':v, 'actual':actual, 'first':first, 'second':second,
                       'first_error':actual-first, 'second_error':actual-second})
    result = {'description':'finite synthetic quadratic experiment; no neural-network claim',
              'bounds':bounds, 'initial_gradient':gradient(p,h).tolist(),
              'runs':summaries, 'point_curvature_counterexample': {'eta':1.0, 'H_at_origin':[[1,0],[0,1]],
              'initial_loss':0.0, 'new_point':[[-1.0,0.0]], 'new_loss':steepening([[-1.0, 0.0]])}}
    # Validate and serialize all numerical records before the first filesystem write.
    env = {'python':platform.python_version(), 'numpy':np.__version__, 'device':'CPU', 'randomness':'none; seed is reserved metadata', 'platform_family':platform.system()}
    json.dumps({'result':result,'trajectories':records,'taylor':taylor,'environment':env}, allow_nan=False)
    payloads = {'results.json':json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                'environment.json':json.dumps(env, indent=2, allow_nan=False)+'\n'}
    for filename, rows in [('trajectories.csv', records), ('taylor.csv', taylor)]:
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
        payloads[filename] = buffer.getvalue()
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    for filename, contents in payloads.items():
        (out/filename).write_text(contents, encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'data/config.json')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs')
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
