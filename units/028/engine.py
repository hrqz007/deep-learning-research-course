"""First-order real float64 autodiff with explicit broadcast adjoints.

No framework or general Jacobian construction. Supported operations: elementwise
add/multiply/negate, tanh, ReLU, sum and an explicit gradient barrier detach.
"""
from dataclasses import dataclass, field
import math
import numpy as np

LIMIT=1e100
MAX_SIZE=100000

def _no_bool(value):
    if isinstance(value,(bool,np.bool_)):raise ValueError('boolean is not a numeric coordinate')
    if isinstance(value,(list,tuple)):
        for v in value:_no_bool(v)
    if isinstance(value,np.ndarray) and value.dtype.kind=='b':raise ValueError('boolean array')


def numeric(value,name='array'):
    _no_bool(value)
    try:a=np.asarray(value)
    except (ValueError,TypeError) as exc:raise ValueError(name+': rectangular real array required') from exc
    if a.dtype.kind not in 'iuf' or a.ndim>4 or a.size==0 or a.size>MAX_SIZE:
        raise ValueError(name+': real scalar or nonempty array, rank<=4, size<=100000')
    if not np.isfinite(a).all() or np.any(np.abs(a)>LIMIT):raise ValueError(name+': finite magnitude<=1e100 required')
    with np.errstate(under='ignore'):b=a.astype(np.float64,copy=True)
    if np.any((a!=0)&(b==0)):raise ValueError(name+': nonzero input underflows in float64 conversion')
    return b


def _broadcast_shape(a,b):
    try:shape=np.broadcast_shapes(a.shape,b.shape)
    except ValueError as exc:raise ValueError('incompatible broadcast shapes') from exc
    if math.prod(shape)>MAX_SIZE:raise ValueError('broadcast result exceeds100000 coordinates')
    return shape


def multiply(a,b):
    a=numeric(a,'left factor');b=numeric(b,'right factor');_broadcast_shape(a,b)
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):out=a*b
    result=numeric(out,'product')
    if np.any((a!=0)&(b!=0)&(result==0)):raise ValueError('nonzero product underflows to zero')
    return result


def sum_to_shape(gradient,shape):
    """Adjoint of broadcasting: add each copied position back to its source."""
    g=numeric(gradient,'incoming gradient')
    if type(shape) is not tuple or any(type(v) is not int or v<1 for v in shape) or len(shape)>g.ndim:
        raise ValueError('valid nonempty-axis source shape required; scalar uses()')
    padded=(1,)*(g.ndim-len(shape))+shape
    if any(s!=1 and s!=t for s,t in zip(padded,g.shape)):raise ValueError('shape could not have broadcast to gradient shape')
    with np.errstate(over='ignore',invalid='ignore'):
        while g.ndim>len(shape):g=g.sum(axis=0)
        for axis,size in enumerate(shape):
            if size==1 and g.shape[axis]!=1:g=g.sum(axis=axis,keepdims=True)
    return numeric(g.reshape(shape),'unbroadcasted gradient')


@dataclass(frozen=True,eq=False,slots=True,repr=False)
class Tensor:
    # Ask NumPy's left-hand array operators to defer to our reflected methods.
    __array_priority__ = 1000
    _value: object
    _parents: tuple=field(default=(),repr=False)
    name: str=''
    op: str='leaf'

    def __post_init__(self):
        value=numeric(self._value,'Tensor value');value.setflags(write=False)
        object.__setattr__(self,'_value',value)
        if type(self._parents) is not tuple or type(self.name) is not str or type(self.op) is not str:raise ValueError('invalid node metadata')
        for edge in self._parents:
            if type(edge) is not tuple or len(edge)!=2 or not isinstance(edge[0],Tensor) or not callable(edge[1]):raise ValueError('invalid pullback edge')

    @property
    def value(self):
        # Defensive copy: even re-enabling writes on this copy cannot change the graph.
        out=self._value.copy();out.setflags(write=False);return out

    @property
    def shape(self):return self._value.shape

    def __repr__(self):return f'Tensor(shape={self.shape}, op={self.op!r}, name={self.name!r})'

    def __add__(self,other):
        other=as_tensor(other);_broadcast_shape(self._value,other._value)
        with np.errstate(over='ignore',invalid='ignore'):value=self._value+other._value
        return Tensor(value,((self,lambda g:sum_to_shape(g,self.shape)),
                             (other,lambda g:sum_to_shape(g,other.shape))),op='+')

    def __mul__(self,other):
        other=as_tensor(other);value=multiply(self._value,other._value)
        return Tensor(value,((self,lambda g:sum_to_shape(multiply(g,other._value),self.shape)),
                             (other,lambda g:sum_to_shape(multiply(g,self._value),other.shape))),op='*')

    def __neg__(self):return Tensor(-self._value,((self,lambda g:-g),),op='neg')
    def __sub__(self,other):return self+(-as_tensor(other))
    def __rsub__(self,other):return as_tensor(other)+(-self)
    def __radd__(self,other):return self+other
    def __rmul__(self,other):return self*other

    def tanh(self):
        if np.any(np.abs(self._value)>100):raise ValueError('tanh requires |input|<=100')
        q=np.exp(-2*np.abs(self._value));slope=4*q/(1+q)**2
        slope.setflags(write=False)
        return Tensor(np.tanh(self._value),((self,lambda g:multiply(g,slope)),),op='tanh')

    def relu(self):
        slope=(self._value>0).astype(float);slope.setflags(write=False)
        return Tensor(np.maximum(self._value,0),((self,lambda g:multiply(g,slope)),),op='relu')

    def sum(self,axis=None,keepdims=False):
        if type(keepdims) is not bool:raise ValueError('keepdims must be bool')
        rank=self._value.ndim
        if axis is None:axes=tuple(range(rank))
        elif type(axis) is int:axes=(axis,)
        elif type(axis) is tuple:axes=axis
        else:raise ValueError('axis must be None, int or tuple of ints')
        if any(type(a) is not int or not -rank<=a<rank for a in axes):raise ValueError('invalid reduction axis')
        axes=tuple(sorted(a%rank for a in axes))
        if len(set(axes))!=len(axes):raise ValueError('duplicate reduction axis')
        with np.errstate(over='ignore',invalid='ignore'):value=self._value.sum(axis=axes,keepdims=keepdims)
        def pullback(g):
            if not keepdims:
                for a in axes:g=np.expand_dims(g,a)
            return numeric(np.broadcast_to(g,self.shape),'sum pullback')
        return Tensor(value,((self,pullback),),op='sum')

    def detach(self):
        """Same forward numbers, a fresh independent leaf: an explicit derivative barrier."""
        return Tensor(self._value,name=self.name+'_detached')


def as_tensor(value):return value if isinstance(value,Tensor) else Tensor(value)


def topological(root):
    if not isinstance(root,Tensor):raise ValueError('Tensor root required; numeric gradients are not derivative graphs')
    state={};order=[];stack=[(root,False)]
    while stack:
        node,done=stack.pop()
        if done:state[node]=2;order.append(node);continue
        if state.get(node)==2:continue
        if state.get(node)==1:raise ValueError('cyclic graph')
        state[node]=1
        if len(state)>10000:raise ValueError('graph exceeds10000 nodes')
        stack.append((node,True))
        for parent,_ in reversed(node._parents):
            if state.get(parent)==1:raise ValueError('cyclic graph')
            if state.get(parent)!=2:stack.append((parent,False))
    return order


def vjp(root,seed=None):
    """Return J^T seed as a fresh dictionary, with each input's original shape.

    Only a true rank-zero scalar root may omit the seed. Vector roots require
    an explicit same-shape cotangent, including a one-element vector root.
    """
    order=topological(root)
    if seed is None:
        if root.shape!=():raise ValueError('non-scalar output requires explicit same-shape seed')
        seed=np.array(1.)
    seed=numeric(seed,'seed')
    if seed.shape!=root.shape:raise ValueError('seed shape must exactly match output; no seed broadcasting')
    grads={node:np.zeros(node.shape,dtype=np.float64) for node in order};grads[root]=seed.copy()
    for node in reversed(order):
        for parent,pullback in node._parents:
            contribution=numeric(pullback(grads[node]),'pullback result')
            if contribution.shape!=parent.shape:raise ValueError('pullback shape differs from parent')
            with np.errstate(over='ignore',invalid='ignore'):new=grads[parent]+contribution
            grads[parent]=numeric(new,'gradient accumulation')
    return grads
