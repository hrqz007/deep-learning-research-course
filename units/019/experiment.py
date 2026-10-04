"""Finite entropy / cross-entropy / KL teaching experiment, Python stdlib only.

Synthetic data; no training API, network access, or nonstandard dependencies.
All objectives use natural logarithms unless base=2 is explicitly requested.
"""
from pathlib import Path
import argparse
import csv
import io
import json
import math
from numbers import Real

ROOT = Path(__file__).resolve().parent
TOL = 1e-12
MAX_STATES = 64


def distribution(values):
    """Validate and canonicalize a finite probability vector.

    Only rounding-level total error <=1e-12 is accepted and divided out.
    Materially invalid weights are rejected, never silently normalized.
    Exact input zeros stay zero. Positive values lost in float conversion fail.
    """
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= MAX_STATES:
        raise ValueError('probabilities must be a list/tuple of 1..64 states')
    out = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError('probabilities must be finite real numbers, not booleans')
        if value < 0 or value > 1:
            raise ValueError('every probability must be in [0,1]')
        v = float(value)
        if not math.isfinite(v) or (value > 0 and v == 0):
            raise ValueError('nonfinite or underflowed probability')
        out.append(v)
    total = math.fsum(out)
    if abs(total - 1.0) > TOL:
        raise ValueError('probability total must be 1 within 1e-12')
    return tuple(v / total for v in out)


def _pair(p, q):
    p, q = distribution(p), distribution(q)
    if len(p) != len(q):
        raise ValueError('p and q must use the same ordered state space')
    return p, q


def _log_scale(base):
    if isinstance(base, bool) or not isinstance(base, Real):
        raise ValueError('base must be a finite real number > 1')
    try:
        value = float(base)
    except (OverflowError, ValueError):
        raise ValueError('base must be representable as a finite float > 1') from None
    if not math.isfinite(value) or value <= 1:
        raise ValueError('base must be a finite real number > 1')
    return math.log(value)


def entropy(p, base=math.e):
    p = distribution(p)
    scale = _log_scale(base)
    return math.fsum(-x * math.log(x) for x in p if x > 0) / scale


def cross_entropy(p, q, base=math.e):
    p, q = _pair(p, q)
    scale = _log_scale(base)
    if any(x > 0 and y == 0 for x, y in zip(p, q)):
        return math.inf
    return math.fsum(-x * math.log(y) for x, y in zip(p, q) if x > 0) / scale


def kl_terms(p, q, base=math.e):
    """Individual terms may be negative; zero mass skips all log operations."""
    p, q = _pair(p, q)
    scale = _log_scale(base)
    terms = []
    for x, y in zip(p, q):
        if x == 0:
            terms.append(0.0)
        elif y == 0:
            terms.append(math.inf)
        else:
            # log differences avoid intermediate x/y overflow or underflow.
            terms.append(x * (math.log(x) - math.log(y)) / scale)
    return tuple(terms)


def kl(p, q, base=math.e):
    terms = kl_terms(p, q, base)
    if any(math.isinf(t) for t in terms):
        return math.inf
    # Do not conceal rounding-level negatives by clipping to zero.
    return math.fsum(terms)


def empirical_distribution(labels, states):
    if isinstance(states, bool) or not isinstance(states, int) or not 1 <= states <= MAX_STATES:
        raise ValueError('states must be an integer in 1..64')
    if not isinstance(labels, (list, tuple)) or not 1 <= len(labels) <= 100000:
        raise ValueError('labels must be a nonempty list/tuple of at most 100000 items')
    counts = [0] * states
    for label in labels:
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < states:
            raise ValueError('every label must be an integer state index')
        counts[label] += 1
    return tuple(c / len(labels) for c in counts)


def mean_nll(labels, q):
    q = distribution(q)
    # Use the shared validation, then independently average per-observation losses.
    empirical_distribution(labels, len(q))
    if any(q[y] == 0 for y in labels):
        return math.inf
    return math.fsum(-math.log(q[y]) for y in labels) / len(labels)


def mix_with_uniform(q, alpha):
    q = distribution(q)
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not 0 <= alpha <= 1 or not math.isfinite(float(alpha)):
        raise ValueError('alpha must be a finite real number in [0,1]')
    a = float(alpha)
    if alpha > 0 and a == 0:
        raise ValueError('positive alpha underflows in float conversion')
    share = a/len(q)
    if a > 0 and share == 0:
        raise ValueError('uniform mixing share underflows; cannot preserve positive support')
    mixed = [(1-a)*x + share for x in q]
    if a > 0 and any(x <= 0 for x in mixed):
        raise ValueError('positive mixed probability underflows')
    return distribution(mixed)


def load_distributions(path=ROOT / 'data/distributions.csv'):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['name','A','B','C']:
            raise ValueError('exact unique CSV header name,A,B,C required')
        rows = list(reader)
    if not rows or any(set(row) != {'name','A','B','C'} or any(value is None or value == '' for value in row.values()) for row in rows):
        raise ValueError('expected CSV header name,A,B,C and at least one row')
    result = {}
    for row in rows:
        if not row['name'] or row['name'] in result:
            raise ValueError('distribution names must be nonempty and unique')
        result[row['name']] = distribution([float(row[k]) for k in ['A','B','C']])
    return result


def load_labels(path=ROOT / 'data/labels.csv'):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['position','label']:
            raise ValueError('exact unique CSV header position,label required')
        rows = list(reader)
    if not rows or any(set(row) != {'position','label'} or any(value is None or value == '' for value in row.values()) for row in rows):
        raise ValueError('expected CSV header position,label and at least one row')
    if [r['position'] for r in rows] != [str(i) for i in range(1,len(rows)+1)]:
        raise ValueError('positions must be consecutive from 1')
    if any(r['label'] not in ('0','1','2') for r in rows):
        raise ValueError('labels must be exact 0,1,2 strings')
    labels = [int(r['label']) for r in rows]
    empirical_distribution(labels, 3)
    return labels


def _json_safe(value):
    if isinstance(value, float) and math.isinf(value):
        return 'Infinity' if value > 0 else '-Infinity'
    if isinstance(value, dict):
        return {k:_json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [_json_safe(v) for v in value]
    return value


def run(output_dir=ROOT / 'outputs'):
    output_dir = Path(output_dir)
    ds = load_distributions()
    required = {'main_p','main_q','zero_p','zero_q','target','broad','left','right'}
    if not required.issubset(ds):
        raise ValueError('missing required distributions: '+','.join(sorted(required-set(ds))))
    p, q = ds['main_p'], ds['main_q']
    labels = load_labels()
    directional = []
    for name in ['broad', 'left', 'right']:
        candidate = ds[name]
        directional.append({'candidate':name,'forward':kl(ds['target'],candidate),'reverse':kl(candidate,ds['target'])})
    eps_rows = []
    for exponent in range(1, 13):
        eps = 10.0 ** -exponent
        eps_rows.append({'epsilon':eps,'cross_entropy':cross_entropy([.5,.5],[1-eps,eps]),'forward':kl([.5,.5],[1-eps,eps]),'reverse':kl([1-eps,eps],[.5,.5])})
    smoothed=mix_with_uniform(ds['zero_q'],.02)
    report={
        'unit':'019','data_kind':'original synthetic finite distributions','log_unit':'nat',
        'main':{'entropy':entropy(p),'cross_entropy':cross_entropy(p,q),'kl':kl(p,q),'reverse_kl':kl(q,p),'kl_terms':kl_terms(p,q),'entropy_bits':entropy(p,2),'cross_entropy_bits':cross_entropy(p,q,2),'kl_bits':kl(p,q,2)},
        'asymmetry':{'forward':kl([.75,.25],[.5,.5]),'reverse':kl([.5,.5],[.75,.25])},
        'empirical':{'labels':labels,'p_hat':empirical_distribution(labels,3),'mean_nll':mean_nll(labels,q),'empirical_cross_entropy':cross_entropy(empirical_distribution(labels,3),q)},
        'zeros':{'forward':kl(ds['zero_p'],ds['zero_q']),'reverse':kl(ds['zero_q'],ds['zero_p']),'cross_entropy':cross_entropy(ds['zero_p'],ds['zero_q']),'common_zero_kl':kl([1,0],[1,0])},
        'smoothed':{'alpha':.02,'q':smoothed,'cross_entropy':cross_entropy(ds['zero_p'],smoothed)},
        'directional':directional,
        'best_candidates':{'forward':[r['candidate'] for r in directional if abs(r['forward']-min(t['forward'] for t in directional))<1e-14],'reverse':[r['candidate'] for r in directional if abs(r['reverse']-min(t['reverse'] for t in directional))<1e-14]},
        'nll_examples':{'q_true_0.6':-math.log(.6),'q_true_0.99':-math.log(.99),'wrong_q_true_0.01':-math.log(.01)},
        'rare_unseen':{'p_rare':.001,'n':100,'probability_no_rare':.999**100,'true_cross_entropy':'Infinity','observed_mean_nll_if_no_rare':0.0},
        'numerics':{'tiny_positive':1e-300,'finite_loss':cross_entropy([0,1],[1-1e-300,1e-300]),'direct_ratio_example':kl([1e-300,1],[1,1e-300])},
    }
    def serialized_csv(rows, fields):
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
        return stream.getvalue()
    # Validate and compute everything before creating or changing output files.
    payload = {
        'directional.csv': serialized_csv(directional,['candidate','forward','reverse']),
        'epsilon.csv': serialized_csv(eps_rows,['epsilon','cross_entropy','forward','reverse']),
        'summary.json': json.dumps(_json_safe(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n',
    }
    output_dir.mkdir(parents=True,exist_ok=True)
    for name,text in payload.items():
        (output_dir/name).write_text(text,encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    args=parser.parse_args()
    report=run(args.output_dir)
    print(json.dumps(_json_safe({'main':report['main'],'best_candidates':report['best_candidates'],'zeros':report['zeros']}),ensure_ascii=False,allow_nan=False))

if __name__=='__main__':
    main()
