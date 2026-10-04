"""031: fixed CPU diagnostic experiments, not a model-quality benchmark.

Public callable helpers accept bounded dense real arrays. The CLI intentionally
accepts only byte-identical teaching inputs; explore through helpers in a new
output directory. No downloads, installs, GPU, external datasets or other units.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import math
import numbers
import os
import tempfile
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
INPUT_SHA = {'batch.csv': '0fa0ff7f91b8d8760681cd2daba96e123d90a17ed4371b541e3a04c195430267', 'config.json': 'd81b6f450fbc363b7d4ed93fccccacf64ca2ced00e3654a251e7f6f803eeb1d8'}
OUTPUT_NAMES = ('summary.json', 'finite-difference.csv', 'training.csv',
                'faults.json', 'perturbations.json', 'fitted-values.csv', 'full-trace.json')


def bounded_array(value, name, shape=None):
    # Check scalar types BEFORE coercion: mixed bool/float lists must not hide booleans.
    objects = np.asarray(value, dtype=object)
    if objects.size == 0:
        raise ValueError(name + ': empty input')
    for scalar in objects.flat:
        if isinstance(scalar, (bool, np.bool_)) or not isinstance(scalar, numbers.Real):
            raise ValueError(name + ': real non-boolean numeric values required')
        if not -1000 <= scalar <= 1000:
            raise ValueError(name + ': nonfinite or magnitude > 1000')
        if scalar != 0 and float(scalar) == 0:
            raise ValueError(name + ': nonzero underflow')
    result = np.array(value, dtype=np.float64, copy=True)
    if shape is not None and result.shape != shape:
        raise ValueError(name + ': unexpected shape')
    return result


def validate_batch(X, y):
    X = bounded_array(X, 'X')
    if X.ndim != 2 or X.shape[1] != 2 or not 1 <= len(X) <= 64:
        raise ValueError('X must have shape (n,2), 1 <= n <= 64')
    y = bounded_array(y, 'y', (len(X), 1))
    return X, y


def scalar_reference(X, y, theta):
    """Independent Python-loop forward and hand-derived chain rule, 13 params."""
    X, y = validate_batch(X, y)
    t = bounded_array(theta, 'theta', (13,)).tolist()
    g = [0.0] * 13
    loss = 0.0
    for row, target in zip(X.tolist(), y[:, 0].tolist()):
        h = [math.tanh(row[0]*t[j]+row[1]*t[3+j]+t[6+j]) for j in range(3)]
        pred = sum(h[j]*t[9+j] for j in range(3)) + t[12]
        residual = pred - target
        loss += residual*residual/(2*len(X))
        upstream = residual/len(X)
        for j in range(3):
            a = upstream*t[9+j]*(1-h[j]*h[j])
            g[j] += a*row[0]
            g[3+j] += a*row[1]
            g[6+j] += a
            g[9+j] += upstream*h[j]
        g[12] += upstream
    return loss, np.array(g)


def flat_loss(theta, X, y, detach_hidden=False):
    """Internal tensor expression: theta(13,), X(n,2), y(n,1)."""
    h = torch.tanh(X @ theta[:6].reshape(2, 3) + theta[6:9])
    if detach_hidden:
        h = h.detach()
    pred = h @ theta[9:12].reshape(3, 1) + theta[12]
    return 0.5*(pred-y).square().mean()


def autograd_values(X, y, theta, detach_hidden=False):
    X, y = validate_batch(X, y)
    theta = bounded_array(theta, 'theta', (13,))
    if type(detach_hidden) is not bool:
        raise ValueError('detach_hidden must be bool')
    tx, ty = torch.tensor(X), torch.tensor(y)
    t = torch.tensor(theta, requires_grad=True)
    loss = flat_loss(t, tx, ty, detach_hidden)
    g = torch.autograd.grad(loss, t)[0]
    return float(loss.detach()), g.detach().numpy().copy()


def central_difference(X, y, theta, step=1e-6):
    X, y = validate_batch(X, y)
    t = bounded_array(theta, 'theta', (13,))
    if isinstance(step, (bool, np.bool_)) or not isinstance(step, (float, int)):
        raise ValueError('step must be a real Python scalar')
    if not math.isfinite(step) or not 1e-12 <= step <= .1:
        raise ValueError('step must lie in [1e-12, 0.1]')
    result = np.empty(13)
    for k in range(13):
        plus, minus = t.copy(), t.copy()
        plus[k] += step
        minus[k] -= step
        result[k] = (scalar_reference(X, y, plus)[0] - scalar_reference(X, y, minus)[0])/(2*step)
    return result


def fd_report(X, y, theta, steps):
    _, g = autograd_values(X, y, theta)
    rows = []
    for step in steps:
        numeric = central_difference(X, y, theta, step)
        for k, (a, n) in enumerate(zip(g, numeric)):
            tol = 1e-8 + 1e-5*max(abs(a), abs(n))
            rows.append({'step':step, 'coordinate':k, 'autograd':float(a),
                         'finite_difference':float(n), 'abs_error':float(abs(a-n)),
                         'tolerance':float(tol), 'pass':bool(abs(a-n) <= tol)})
    return rows


class Tiny(nn.Module):
    """2 -> 12 -> 1 tanh; registered parameters, local deterministic generator."""
    def __init__(self, seed=31):
        super().__init__()
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError('seed must be an integer in [0,2**31)')
        generator = torch.Generator(device='cpu').manual_seed(seed)
        self.W1 = nn.Parameter(.5*torch.randn(2, 12, generator=generator, dtype=torch.float64))
        self.b1 = nn.Parameter(.1*torch.randn(12, generator=generator, dtype=torch.float64))
        self.W2 = nn.Parameter(.3*torch.randn(12, 1, generator=generator, dtype=torch.float64))
        self.b2 = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, x, detach_hidden=False):
        h = torch.tanh(x @ self.W1 + self.b1)
        if detach_hidden:
            h = h.detach()
        return h @ self.W2 + self.b2


def norm(t):
    return float(torch.linalg.vector_norm(t.detach()))


def train_one_batch(X, y, steps=1600, lr=.03, seed=31):
    X, y = validate_batch(X, y)
    if type(steps) is not int or not 1 <= steps <= 5000:
        raise ValueError('steps must be an integer in [1,5000]')
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or not 0 < lr <= .1:
        raise ValueError('learning rate must be in (0,0.1]')
    model = Tiny(seed)
    model.train()
    tx, ty = torch.tensor(X), torch.tensor(y)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    trace = []
    for step in range(steps+1):
        opt.zero_grad(set_to_none=True)
        loss = .5*(model(tx)-ty).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('nonfinite training loss')
        loss.backward()
        grads = [p.grad for p in model.parameters()]
        if any(g is None or not bool(torch.isfinite(g).all()) for g in grads):
            raise ValueError('missing or nonfinite training gradient')
        gnorm = math.sqrt(sum(norm(g)**2 for g in grads))
        before = [p.detach().clone() for p in model.parameters()]
        if step < steps:
            opt.step()
        delta = math.sqrt(sum(norm(p-old)**2 for p, old in zip(model.parameters(), before)))
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise ValueError('nonfinite training parameter')
        trace.append({'step':step, 'loss_before_step':float(loss.detach()),
                      'gradient_norm':gnorm, 'update_norm':delta})
    return model, trace


def one_step_probe(X, y, mode):
    """Identical initial state, plain SGD; no historical optimizer state."""
    X, y = validate_batch(X, y)
    modes = ('healthy', 'no_step', 'zero_lr', 'omit_W1', 'detach_hidden', 'sum_reduction')
    if mode not in modes:
        raise ValueError('unknown fault mode')
    model = Tiny()
    parameters = [(name, p) for name, p in model.named_parameters()]
    included = [p for name, p in parameters if not (mode == 'omit_W1' and name == 'W1')]
    opt = torch.optim.SGD(included, lr=0 if mode == 'zero_lr' else .03)
    # Reset every model parameter, even ones omitted from the optimizer.
    model.zero_grad(set_to_none=True)
    pred = model(torch.tensor(X), detach_hidden=(mode == 'detach_hidden'))
    residual = pred-torch.tensor(y)
    loss = .5*(residual.square().sum() if mode == 'sum_reduction' else residual.square().mean())
    loss.backward()
    before = {name:p.detach().clone() for name,p in parameters}
    if mode != 'no_step':
        opt.step()
    rows = []
    for name, p in parameters:
        rows.append({'name':name, 'in_optimizer':any(p is q for q in included),
                     'grad_is_none':p.grad is None,
                     'grad_norm':None if p.grad is None else norm(p.grad),
                     'update_norm':norm(p-before[name])})
    return {'loss':float(loss.detach()), 'parameters':rows}


def fault_report(X, y, theta):
    good_loss, g = autograd_values(X, y, theta)
    numeric = central_difference(X, y, theta)
    cut_loss, cut_g = autograd_values(X, y, theta, True)
    tx, ty = torch.tensor(X), torch.tensor(y)
    t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    ordinary_pass = bool(torch.autograd.gradcheck(lambda q:flat_loss(q, tx, ty), (t,), eps=1e-6, atol=1e-8, rtol=1e-5))
    cut_pass = bool(torch.autograd.gradcheck(lambda q:flat_loss(q, tx, ty, True), (t,), eps=1e-6, atol=1e-8, rtol=1e-5, raise_exception=False))
    permuted_pass = bool(torch.autograd.gradcheck(lambda q:flat_loss(q, tx, ty.flip(0)), (t,), eps=1e-6, atol=1e-8, rtol=1e-5))
    w = torch.tensor(-1., dtype=torch.float64, requires_grad=True)
    unused = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    connected, absent = torch.autograd.grad(torch.relu(w), (w, unused), allow_unused=True)
    kink = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    kink_g = float(torch.autograd.grad(torch.relu(kink), kink)[0])
    # Orthogonal error is invisible to one direction, even with exact arithmetic.
    true_g = np.array([1., 2.]); wrong_g = np.array([2., 1.]); v = np.array([1., 1.])/math.sqrt(2)
    one_steps = {mode:one_step_probe(X, y, mode) for mode in ('healthy','no_step','zero_lr','omit_W1','detach_hidden','sum_reduction')}
    pred = torch.tensor([[1.], [2.]], dtype=torch.float64)
    target = torch.tensor([[1.], [2.]], dtype=torch.float64)
    report = {'gradcheck':{'healthy':ordinary_pass, 'detached_hidden':cut_pass,
                          'permuted_labels':permuted_pass},
              'gradient_compare':{'loss':good_loss, 'cut_loss':cut_loss,
                  'healthy_max_abs':float(np.max(np.abs(g-numeric))),
                  'cut_max_abs':float(np.max(np.abs(cut_g-numeric))),
                  'cut_failed_coordinates':[int(k) for k in np.flatnonzero(np.abs(cut_g-numeric)>1e-7)],
                  'missing_mean_factor_max_abs':float(np.max(np.abs(len(X)*g-numeric)))},
              'none_vs_zero':{'connected_relu_gradient':float(connected),'unused_gradient_is_none':absent is None},
              'relu_at_zero':{'autograd':kink_g,'central_difference':.5,'ordinary_derivative_exists':False},
              'direction_blind_spot':{'true_dot':float(true_g@v),'wrong_dot':float(wrong_g@v),'coordinate_error_norm':float(np.linalg.norm(wrong_g-true_g))},
              'broadcast':{'correct_shape':list((pred-target).shape),'incorrect_shape':list((pred-target[:,0]).shape),
                           'correct_loss':float(.5*(pred-target).square().mean()),'incorrect_loss':float(.5*(pred-target[:,0]).square().mean())},
              'contradictory_duplicate':{'same_input':[0.,0.],'targets':[-1.,1.],'optimal_prediction':0.,'minimum_half_mse':.5},
              'one_step':one_steps}
    if not ordinary_pass or cut_pass or not permuted_pass:
        raise ValueError('unexpected gradcheck diagnostic result')
    return report


def perturbation_report(model, X, y):
    X, y = validate_batch(X, y)
    tx, ty = torch.tensor(X), torch.tensor(y)
    model.eval()  # No Dropout/BatchNorm here; held fixed, not a new training run.
    with torch.no_grad():
        base = model(tx)
        def loss(pred, target):return float(.5*(pred-target).square().mean())
        rows=[]
        for eps in (.1, .01, .001):
            shift=torch.zeros_like(tx); shift[:,0]=eps
            shifted=model(tx+shift)
            # Target shift is known from the synthetic generating equation.
            rows.append({'epsilon':eps,'mean_abs_prediction_change':float((shifted-base).abs().mean()),
                         'mean_slope':float(((shifted-base)/eps).mean()),
                         'known_rule_slope':.7,'loss_against_shifted_rule':loss(shifted,ty+.7*eps)})
        order=torch.arange(len(X)-1,-1,-1)
        shuffled=model(tx[order])
        zero=model(torch.zeros_like(tx))
        return {'fixed_model':True,'baseline_loss':loss(base,ty),
                'joint_row_permutation_loss':loss(shuffled,ty[order]),
                'input_only_permutation_loss':loss(shuffled,ty),
                'zero_input_loss_original_targets':loss(zero,ty),
                'mean_target_constant_loss':loss(ty.mean().expand_as(ty),ty),
                'small_shifts':rows,
                'scope':'synthetic interventions on eight training inputs; no robustness or generalization estimate'}


def full_trace(X, y, theta, eta=.1):
    """Every value/adjoint in the same 13-coordinate FD example, then one SGD step."""
    X,y=validate_batch(X,y)
    t=bounded_array(theta,'theta',(13,))
    if isinstance(eta,bool) or not isinstance(eta,(float,int)) or not math.isfinite(eta) or not 0 < eta <= .1:
        raise ValueError('eta must be in (0,0.1]')
    W1=t[:6].reshape(2,3);b1=t[6:9];W2=t[9:12].reshape(3,1);b2=t[12]
    A=X@W1+b1;H=np.tanh(A);prediction=H@W2+b2;residual=prediction-y
    per_sample=.5*residual**2
    upstream=residual/len(X)
    local_tanh=1-H**2
    dH=upstream@W2.T;dA=dH*local_tanh
    dW1=X.T@dA;db1=dA.sum(axis=0);dW2=H.T@upstream;db2=upstream.sum()
    dX=dA@W1.T
    g=np.concatenate([dW1.reshape(-1),db1,dW2.reshape(-1),np.array([db2])])
    contributions=np.column_stack([
        X[:,0:1]*dA, X[:,1:2]*dA, dA, H*upstream, upstream])
    after=t-eta*g
    A_after=X@after[:6].reshape(2,3)+after[6:9]
    H_after=np.tanh(A_after)
    prediction_after=H_after@after[9:12].reshape(3,1)+after[12]
    residual_after=prediction_after-y
    loss_after=float(.5*np.mean(residual_after**2))
    tx=torch.tensor(X,requires_grad=True);ty=torch.tensor(y);tt=torch.tensor(t,requires_grad=True)
    ta=tx@tt[:6].reshape(2,3)+tt[6:9];th=torch.tanh(ta)
    tp=th@tt[9:12].reshape(3,1)+tt[12];tl=.5*(tp-ty).square().mean()
    tg,txg,tag,thg,tpg=torch.autograd.grad(tl,(tt,tx,ta,th,tp))
    manual={'A':A,'H':H,'prediction':prediction,'dA':dA,'dH':dH,'dPrediction':upstream,'gradient':g,'dX':dX}
    actual={'A':ta,'H':th,'prediction':tp,'dA':tag,'dH':thg,'dPrediction':tpg,'gradient':tg,'dX':txg}
    comparisons=[]
    for name,value in manual.items():
        other=actual[name].detach().numpy()
        for index in np.ndindex(value.shape):
            difference=abs(float(value[index])-float(other[index]))
            if difference > 1e-12:
                raise ValueError('full trace mismatch: '+name)
            comparisons.append({'tensor':name,'index':list(index),'manual':float(value[index]),'torch':float(other[index]),'abs_error':difference})
    tensors={'X':X,'y':y,'W1':W1,'b1':b1,'W2':W2,'b2':np.array(b2),
             'A':A,'H':H,'prediction':prediction,'residual':residual,'per_sample_half_squared_error':per_sample,
             'dPrediction':upstream,'local_tanh_derivative':local_tanh,'dH':dH,'dA':dA,
             'dW1':dW1,'db1':db1,'dW2':dW2,'db2':np.array(db2),'dX':dX,
             'theta_before':t,'gradient':g,'parameter_contributions':contributions,'theta_after':after,
             'A_after':A_after,'H_after':H_after,'prediction_after':prediction_after,'residual_after':residual_after}
    return {'batch_size':len(X),'eta':eta,'loss_before':float(per_sample.mean()),'loss_after':loss_after,
            'shapes':{name:list(value.shape) for name,value in tensors.items()},
            'tensors':{name:value.tolist() for name,value in tensors.items()},
            'comparisons':comparisons,'comparison_max_abs':max(row['abs_error'] for row in comparisons)}


def encode(value):return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'

def csv_text(rows):
    out=io.StringIO(newline='')
    writer=csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return out.getvalue()


def load_inputs(data_dir=HERE/'data'):
    root=Path(data_dir)
    for name, expected in INPUT_SHA.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('fixed teaching input changed: '+name)
    with (root/'batch.csv').open(newline='', encoding='utf-8') as file:
        rows=list(csv.DictReader(file))
    cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    X=[[float(r['x1']),float(r['x2'])] for r in rows]
    y=[[float(r['target'])] for r in rows]
    yr=[[float(r['random_target'])] for r in rows]
    X,y=validate_batch(X,y); _,yr=validate_batch(X,yr)
    return X,y,yr,cfg


def run(data_dir=HERE/'data', output=HERE/'outputs'):
    X,y,yr,cfg=load_inputs(data_dir)
    fd=fd_report(X[:4],y[:4],cfg['theta'],cfg['fd_steps'])
    ref_loss, ref_grad=scalar_reference(X[:4],y[:4],cfg['theta'])
    torch_loss, torch_grad=autograd_values(X[:4],y[:4],cfg['theta'])
    faults=fault_report(X[:4],y[:4],cfg['theta'])
    trace13=full_trace(X[:4],y[:4],cfg['theta'])
    _, duplicate_trace=train_one_batch([[0.,0.],[0.,0.]],[[-1.],[1.]],300,.03,cfg['seed'])
    faults['contradictory_duplicate']['actual_final_half_mse']=duplicate_trace[-1]['loss_before_step']
    model,trace=train_one_batch(X,y,cfg['steps'],cfg['learning_rate'],cfg['seed'])
    random_model,random_trace=train_one_batch(X,yr,cfg['steps'],cfg['learning_rate'],cfg['seed'])
    perturb=perturbation_report(model,X,y)
    if trace[-1]['loss_before_step'] >= 1e-4 or random_trace[-1]['loss_before_step'] >= 1e-4:
        raise ValueError('fixed single-batch fit did not reach its diagnostic threshold')
    with torch.no_grad():
        predictions=model(torch.tensor(X)).reshape(-1).tolist()
        random_predictions=random_model(torch.tensor(X)).reshape(-1).tolist()
    fitted=[{'id':f's{i}','target':float(y[i,0]),'fitted':predictions[i],
             'random_target':float(yr[i,0]),'random_fitted':random_predictions[i]} for i in range(len(X))]
    training=[dict(label_set=name,**r) for name,records in [('rule',trace),('random',random_trace)] for r in records]
    summary={'torch_version':torch.__version__,'numpy_version':np.__version__,'device':'cpu','dtype':'float64',
             'trace_coordinates':len(trace13['comparisons']),'trace_max_abs':trace13['comparison_max_abs'],
             'trace_loss_after_sgd':trace13['loss_after'],'fd_parameter_count':13,'fd_coordinates_times_steps':len(fd),
             'reference_loss':ref_loss,'torch_loss':torch_loss,
             'reference_gradient_max_abs':float(np.max(np.abs(ref_grad-torch_grad))),
             'fd_1e_6_max_abs':max(r['abs_error'] for r in fd if r['step']==1e-6),
             'train_parameters':sum(p.numel() for p in model.parameters()),'batch_size':len(X),'updates':cfg['steps'],
             'rule_initial_loss':trace[0]['loss_before_step'],'rule_final_loss':trace[-1]['loss_before_step'],
             'random_initial_loss':random_trace[0]['loss_before_step'],'random_final_loss':random_trace[-1]['loss_before_step'],
             'scope':cfg['scope']}
    payload={'summary.json':encode(summary),'finite-difference.csv':csv_text(fd),'training.csv':csv_text(training),
             'faults.json':encode(faults),'perturbations.json':encode(perturb),'fitted-values.csv':csv_text(fitted),
             'full-trace.json':encode(trace13)}
    # Finish validation, calculation and serialization before touching the output.
    # Per-file replacements are atomic; a multi-file OS crash transaction is not claimed.
    dest=Path(output)
    with tempfile.TemporaryDirectory(prefix='dl031-') as staging:
        for name,text in payload.items():(Path(staging)/name).write_text(text,encoding='utf-8')
        dest.mkdir(parents=True,exist_ok=True)
        for name in OUTPUT_NAMES:
            with tempfile.NamedTemporaryFile(dir=dest,prefix='.031-',delete=False) as file:
                temp=Path(file.name);file.write((Path(staging)/name).read_bytes())
            try:os.replace(temp,dest/name)
            finally:temp.unlink(missing_ok=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=HERE/'data')
    parser.add_argument('--output',type=Path,default=HERE/'outputs')
    args=parser.parse_args()
    torch.set_num_threads(1)
    try:summary=run(args.data_dir,args.output)
    except (ValueError, OSError, RuntimeError, TypeError) as exc:
        parser.exit(2,f'input/calculation error: {exc}\n')
    print(encode(summary),end='')


if __name__=='__main__':main()
