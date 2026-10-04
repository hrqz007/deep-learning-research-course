"""026: immutable scalar nodes and reverse topological chain-rule accumulation.
Core uses only the Python standard library. No framework, array, GPU or service.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from decimal import Decimal
import argparse, csv, hashlib, io, json, math, os

HERE = Path(__file__).resolve().parent
OUTPUT_NAMES = ('summary.json', 'graph.csv', 'edges.csv', 'gradient_check.csv', 'training.csv')
DEFAULT = {'schema': 1, 'x': 2.0, 'y': 0.25, 'w': 0.5, 'b': -0.25,
           'iterations': 40, 'learning_rate': 0.02}
TEACHING_HASHES = {'data/config.json': '62d5846ca471393efe4bacfd34117ce108502f2b9a33239ed2347073ebaa3ae2', 'outputs/edges.csv': 'd99398c6612ca984e0e5d0fc38f5e409c4eb382d4854cbc4eec93ee1132596b4', 'outputs/gradient_check.csv': '550bea14d3433c5639d2dc7fba779e02fb4ee672d6841b51b554c0ec4dc57e5c', 'outputs/graph.csv': 'a47f071fdbc0d3239ff206692f9728730f3ebf9afb024336e4ae3169f1ecf64f', 'outputs/summary.json': '8a71cb43f387aaa11dc1da212a741c51f23d64001a3aa1427085fdfdcff2cb39', 'outputs/training.csv': '2a3549610831dcab00a327ba24f05317cfcbd07862d456dd11f493886c767dd7'}  # Fixed presentation-only guard.


def number(value, name='value'):
    """Deliberately accept built-in int/float only, excluding bool and strings."""
    if type(value) not in (int, float):
        raise ValueError(name + ': built-in int or float required (not bool)')
    try: result = float(value)
    except (ValueError, OverflowError) as exc: raise ValueError(name + ': finite float required') from exc
    if not math.isfinite(result) or abs(result) > 1e100:
        raise ValueError(name + ': finite magnitude at most 1e100 required')
    return result


def product(a, b):
    a = number(a, 'left factor')
    b = number(b, 'right factor')
    value = number(a * b, 'product')
    if a != 0.0 and b != 0.0 and value == 0.0:
        raise ValueError('nonzero product underflowed to zero')
    return value


@dataclass(frozen=True, eq=False, slots=True)
class Scalar:
    """Identity is node identity, even when numbers or names are equal.

    Each parents entry is (input_node, local_partial). Repeated operands are
    repeated edges. Never deduplicate edges. Values and edges cannot be mutated
    through the supported API. Ordinary gradients are returned separately.
    """
    value: float
    parents: tuple = ()
    op: str = 'leaf'
    name: str = ''

    def __post_init__(self):
        object.__setattr__(self, 'value', number(self.value))
        if type(self.parents) is not tuple or type(self.op) is not str or type(self.name) is not str:
            raise ValueError('parents tuple and string labels required')
        for edge in self.parents:
            if type(edge) is not tuple or len(edge) != 2 or not isinstance(edge[0], Scalar):
                raise ValueError('each edge is (Scalar, finite local partial)')
            number(edge[1], 'local partial')

    def __add__(self, other):
        other = as_scalar(other)
        return Scalar(number(self.value + other.value, 'sum'),
                      ((self, 1.0), (other, 1.0)), '+')

    def __mul__(self, other):
        other = as_scalar(other)
        return Scalar(product(self.value, other.value),
                      ((self, other.value), (other, self.value)), '*')

    def __neg__(self):
        return Scalar(-self.value, ((self, -1.0),), 'neg')

    def __sub__(self, other): return self + (-as_scalar(other))
    def __rsub__(self, other): return as_scalar(other) + (-self)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other

    def tanh(self):
        if abs(self.value) > 100: raise ValueError('teaching tanh supports |input| <= 100')
        q = math.exp(-2.0 * abs(self.value))
        slope = 4.0 * q / ((1.0 + q) * (1.0 + q))
        return Scalar(math.tanh(self.value), ((self, slope),), 'tanh')

    def relu(self):
        return Scalar(max(self.value, 0.0), ((self, 1.0 if self.value > 0 else 0.0),), 'relu')


def as_scalar(value):
    return value if isinstance(value, Scalar) else Scalar(value)


def topological(root):
    """Iterative DFS: inputs before users, one visit per node, edge multiplicity kept."""
    if not isinstance(root, Scalar): raise ValueError('scalar root required')
    order, state = [], {}
    stack = [(root, False)]
    while stack:
        node, exiting = stack.pop()
        if exiting:
            state[node] = 2
            order.append(node)
            continue
        if state.get(node) == 2: continue
        if state.get(node) == 1: raise ValueError('cycle in graph')
        state[node] = 1
        if len(state) > 100000: raise ValueError('teaching graph exceeds 100000 nodes')
        stack.append((node, True))
        for parent, _ in reversed(node.parents):
            if state.get(parent) == 1: raise ValueError('cycle in graph')
            if state.get(parent) != 2: stack.append((parent, False))
    return order


def backward(root, seed=1.0):
    """A fresh map each call; seed=1 means d(root)/d(node), never an update."""
    seed = number(seed, 'seed')
    order = topological(root)
    gradients = {node: 0.0 for node in order}
    gradients[root] = seed
    for node in reversed(order):
        for parent, local in node.parents:
            contribution = product(gradients[node], local)
            gradients[parent] = number(gradients[parent] + contribution, 'gradient sum')
    return gradients


def example(x=2.0, y=0.25, w=0.5, b=-0.25):
    """s=(w*x+b)^2+w, L=0.5*(s-y)^2. w occurs on two paths."""
    nodes = {k: Scalar(v, name=k) for k, v in [('x',x),('y',y),('w',w),('b',b)]}
    nodes['m'] = nodes['w'] * nodes['x']
    nodes['a'] = nodes['m'] + nodes['b']
    nodes['h'] = nodes['a'] * nodes['a']
    nodes['s'] = nodes['h'] + nodes['w']
    nodes['e'] = nodes['s'] - nodes['y']
    nodes['q'] = nodes['e'] * nodes['e']
    nodes['L'] = 0.5 * nodes['q']
    return nodes


def direct_loss(x, y, w, b):
    """Ordinary float expression; used for finite differences, no graph calls."""
    a = w*x+b
    return 0.5*(a*a+w-y)**2


def explicit_gradient(x, y, w, b):
    a = w*x+b
    e = a*a+w-y
    return {'w': e*(2*a*x+1), 'b': 2*a*e, 'x': 2*a*w*e, 'y': -e}


def float_token(token):
    d = Decimal(token)
    f = number(float(d), 'JSON number')
    if d != 0 and f == 0: raise ValueError('nonzero JSON number underflows during conversion')
    return f


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def read_config(path):
    def bad_constant(value): raise ValueError('nonfinite JSON token: ' + value)
    if Path(path).stat().st_size > 10000: raise ValueError('config exceeds 10000 bytes')
    cfg = json.loads(Path(path).read_text(encoding='utf-8'), parse_float=float_token,
                     parse_constant=bad_constant, object_pairs_hook=unique_object)
    if type(cfg) is not dict or set(cfg) != set(DEFAULT): raise ValueError('exact config keys required')
    if type(cfg['schema']) is not int or cfg['schema'] != 1: raise ValueError('schema must be integer 1')
    for key in ('x','y','w','b'):
        cfg[key] = number(cfg[key], key)
        if abs(cfg[key]) > 10: raise ValueError(key + ' magnitude must be <= 10')
    if type(cfg['iterations']) is not int or not 0 <= cfg['iterations'] <= 500:
        raise ValueError('iterations must be an integer from 0 to 500')
    cfg['learning_rate'] = number(cfg['learning_rate'], 'learning_rate')
    if not 1e-6 <= cfg['learning_rate'] <= 0.1: raise ValueError('learning_rate must be between 1e-6 and 0.1')
    return cfg


def csv_text(rows, fields):
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return buf.getvalue()


def serialize_run(cfg):
    x,y,w,b = (cfg[k] for k in ('x','y','w','b'))
    nodes = example(x,y,w,b); grads = backward(nodes['L']); order = topological(nodes['L'])
    names = {node: k for k,node in nodes.items()}
    for i,node in enumerate(order): names.setdefault(node, 'internal_' + str(i))
    graph = [{'node':names[n], 'operation':n.op, 'value':n.value, 'gradient':grads[n]}
             for n in order]
    edges = [{'output':names[n], 'input':names[p], 'edge_index':j,
              'local_partial':local, 'upstream':grads[n],
              'contribution':product(grads[n],local)}
             for n in reversed(order) for j,(p,local) in enumerate(n.parents)]
    checks=[]
    for exponent in range(1,13):
        step = 10.0**(-exponent)
        for key in ('w','b'):
            plus = {'x':x,'y':y,'w':w,'b':b}; minus=plus.copy()
            plus[key]+=step;minus[key]-=step
            fd=(direct_loss(**plus)-direct_loss(**minus))/(2*step)
            target=grads[nodes[key]]
            checks.append({'parameter':key,'step':step,'autodiff':target,'finite_difference':fd,
                           'absolute_error':abs(fd-target),
                           'scaled_error':abs(fd-target)/max(1.0,abs(fd),abs(target))})
    history=[]
    for i in range(cfg['iterations']+1):
        # New leaves AND new graph each iteration; the old graph is never reused.
        current=example(x,y,w,b); g=backward(current['L'])
        history.append({'step':i,'w':w,'b':b,'prediction':current['s'].value,
                        'loss':current['L'].value,'gradient_w':g[current['w']],
                        'gradient_b':g[current['b']]})
        if i < cfg['iterations']:
            next_w=number(w-product(cfg['learning_rate'],g[current['w']]),'new w')
            next_b=number(b-product(cfg['learning_rate'],g[current['b']]),'new b')
            w,b=next_w,next_b
    summary = {'config':cfg,'initial':{k: {'value':n.value,'gradient':grads[n]} for k,n in nodes.items()},
               'explicit_gradient':explicit_gradient(x,y,cfg['w'],cfg['b']),
               'node_count':len(order),'edge_count':sum(len(n.parents) for n in order),
               'final':history[-1],
               'semantics':'backward returns a fresh gradient map; it does not update parameters',
               'scope':'one synthetic scalar example, not a statistical evaluation'}
    texts = {'summary.json':json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',
             'graph.csv':csv_text(graph,['node','operation','value','gradient']),
             'edges.csv':csv_text(edges,['output','input','edge_index','local_partial','upstream','contribution']),
             'gradient_check.csv':csv_text(checks,['parameter','step','autodiff','finite_difference','absolute_error','scaled_error']),
             'training.csv':csv_text(history,['step','w','b','prediction','loss','gradient_w','gradient_b'])}
    return texts


def run(config=None, output=None):
    config = Path(config) if config is not None else HERE/'data/config.json'
    output = Path(os.path.abspath(output)) if output is not None else HERE/'outputs'
    cfg=read_config(config)
    # Validate destination before computing, but do not create directories yet.
    if output.is_symlink() or any(p.is_symlink() for p in output.parents):
        raise ValueError('symlink output paths are unsupported')
    if output.exists() and not output.is_dir(): raise ValueError('output must be a directory')
    for name in OUTPUT_NAMES:
        p=output/name
        if p.is_symlink() or (p.exists() and not p.is_file()) or p.resolve() == config.resolve(): raise ValueError('unsafe output target')
    texts=serialize_run(cfg)  # All calculations AND serialization precede writes.
    output.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES: (output/name).write_text(texts[name],encoding='utf-8')
    return json.loads(texts['summary.json'])


def require_teaching_inputs():
    """Do not reuse fixed captions/figures for altered inputs or computed outputs."""
    for name, expected in TEACHING_HASHES.items():
        p=HERE/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise ValueError('fixed teaching file differs: ' + name)
    if not TEACHING_HASHES: raise ValueError('teaching hash ledger not initialized')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'data/config.json')
    parser.add_argument('--output',type=Path,default=HERE/'outputs')
    args=parser.parse_args(); result=run(args.config,args.output)
    print(json.dumps({'initial_loss':result['initial']['L']['value'],
                      'initial_gradient_w':result['initial']['w']['gradient'],
                      'final_loss':result['final']['loss']},allow_nan=False))
