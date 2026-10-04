"""Independent rational/Decimal oracles and failure tests for the SVD lesson."""
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import json
import math
import re
import tempfile
import numpy as np
import experiment as lab

COUNT=0

def rejects(fn, kind=(TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError)):
 global COUNT
 try: fn()
 except kind: COUNT+=1
 else: raise AssertionError('expected explicit rejection')

def near(a,b,rtol=3e-12,atol=2e-12):
 np.testing.assert_allclose(a,b,rtol=rtol,atol=atol)


def decimal_singular_values(a,b,c,d):
 # Independent exact-integer trace/determinant, 80-digit scalar sqrt, no SVD/eig.
 with localcontext() as ctx:
  ctx.prec=80
  tr=Decimal(a*a+b*b+c*c+d*d)
  det=Decimal(a*d-b*c)
  high=((tr+(tr*tr-4*det*det).sqrt())/2).sqrt()
  low=abs(det)/high if high else Decimal(0)
  return [float(high),float(low)]


def main():
 global COUNT
 groups=[]
 W=np.array([[3.,1,0],[1,3,0]])
 U,s,Vt=lab.reduced_svd(W)
 near(s,[4,2]);near((U*s)@Vt,W);near(U.T@U,np.eye(2));near(Vt@Vt.T,np.eye(2))
 near(lab.rank_k(W,1),[[2,2,0],[2,2,0]])
 for k,e in enumerate([math.sqrt(20),2,0]):near(lab.frobenius(W-lab.rank_k(W,k)),e)
 groups.append('fixed rectangular hand SVD, shapes, orthogonality, rank truncations')
 fixtures=0
 for a in range(-2,3):
  for b in range(-2,3):
   for c in [-1,2]:
    for d in [0,3]:
     A=[[a,b],[c,d]]; oracle=decimal_singular_values(a,b,c,d)
     near(lab.reduced_svd(A)[1],oracle)
     near(lab.frobenius(np.array(A)-lab.rank_k(A,1)),oracle[1])
     if a*d-b*c!=0:
      near(lab.condition2(A),oracle[0]/oracle[1],rtol=1e-10)
     fixtures+=1
 groups.append('100 independent 80-digit Decimal 2x2 singular-value and tail oracles')
 fractions=0
 for a,b,c,d in [(2,1,0,3),(1,2,3,5),(-3,2,1,4),(5,-1,2,2)]:
  det=Fraction(a*d-b*c)
  for p in range(-3,4):
   q=p+2
   # Row inverse equation: x=(p*d-q*c, -p*b+q*a)/det.
   expected=[float(Fraction(p*d-q*c)/det),float(Fraction(-p*b+q*a)/det)]
   near(lab.solve_row([[a,b],[c,d]],[[p,q]]),[expected]); fractions+=1
 groups.append('28 independent Fraction nonsymmetric row-system oracles')
 for e in [Fraction(1),Fraction(1,100),Fraction(1,10000),Fraction(1,1000000)]:
  delta=Fraction(1,1000000);r=lab.perturbation_case(float(e),float(delta))
  near(r['relative_x'],float(delta/e));near(r['amplification'],float(1/e));near(r['condition2'],float(1/e))
 near(lab.solve_row(np.diag([1,1e-6]),[[1+1e-6,0]]),[[1+1e-6,0]])
 groups.append('exact rational perturbation amplification and safe direction')
 assert lab.numerical_rank(np.diag([1,1e-8]),1e-6)==1
 assert lab.numerical_rank(np.diag([1,1e-8]),1e-10)==2
 assert lab.numerical_rank(np.diag([1,.5]),.5)==1
 assert lab.numerical_rank(np.zeros((2,3)),0)==0
 assert math.isinf(lab.condition2([[1,0],[0,0]]))
 assert math.isinf(lab.condition2(np.zeros((2,2))))
 near(lab.reduced_svd([[0,1],[0,0]])[1],[1,0])
 near(lab.reduced_svd([[-3,0],[0,1]])[1],[3,1])
 near(lab.rank_k(np.eye(2),1)@lab.rank_k(np.eye(2),1),lab.rank_k(np.eye(2),1))
 groups.append('rank tolerance boundary, zero, singular, negative eigenvalue and repeated spectrum')
 for bad in [True,[],[1,2],[[1],[2,3]],np.empty((0,2)),[[True,1]],[[np.bool_(True),2]],[[1+2j]],[["3"]],[[None]],[[np.nan]],[[np.inf]],np.array([[True]])]:rejects(lambda bad=bad:lab.matrix(bad))
 rejects(lambda:lab.matrix(np.array([[np.longdouble('1e400')]])))
 for k in [-1,3,1.2,True,np.bool_(False),'1']:rejects(lambda k=k:lab.rank_k(W,k))
 for t in [-1,np.inf,np.nan,True,'x',1j]:rejects(lambda t=t:lab.numerical_rank(W,t))
 rejects(lambda:lab.condition2(W));rejects(lambda:lab.solve_row(W,[[1,2,3]]));rejects(lambda:lab.solve_row(np.eye(2),[1,2]));rejects(lambda:lab.solve_row(np.zeros((2,2)),[[1,2]]))
 for e,d in [(0,1),(-1,1),(1,0),(np.nan,1),(True,1),(1,'x'),(1e-308,1e308)]:rejects(lambda e=e,d=d:lab.perturbation_case(e,d))
 rejects(lambda:lab.frobenius([[1e308,1e308],[1e308,1e308]]))
 rejects(lambda:lab.condition2([[1e160,0],[0,1e-160]]))
 before=W.copy();lab.rank_k(W,1);near(W,before)
 groups.append('explicit type/shape/range/overflow rejections and input immutability')
 with tempfile.TemporaryDirectory() as temp:
  p=Path(temp);r=lab.run(p/'a');lab.run(p/'b')
  for f in (p/'a').iterdir():assert f.read_bytes()==(p/'b'/f.name).read_bytes()
  assert r['task_counterexample']['full_output'][0]!=r['task_counterexample']['full_output'][1]
  assert r['task_counterexample']['rank_one_output'][0]==r['task_counterexample']['rank_one_output'][1]
  badcfg=p/'bad.json';badcfg.write_text(json.dumps({'matrix':[[1]],'epsilons':[0],'delta':1e-6}))
  rejects(lambda:lab.run(p/'failed',badcfg));assert not (p/'failed').exists()
  badcfg.write_text(json.dumps({'matrix':[[1]],'epsilons':[],'delta':1e-6}))
  rejects(lambda:lab.run(p/'empty',badcfg));assert not (p/'empty').exists()
 groups.append('repeat byte-identical outputs, information loss, invalid config fails before output')
 namespace={};snippets=0
 for name in ['lecture.md','lab.md','answers.md']:
  for code in re.findall(r'```python\n(.*?)```',(lab.ROOT/name).read_text(),re.S):
   exec(compile(code,name,'exec'),namespace);snippets+=1
 groups.append('all Python document blocks executed in order')
 report={'status':'passed','groups':len(groups),'test_groups':groups,'decimal_fixtures':fixtures,'fraction_solve_fixtures':fractions,'rejected_cases':COUNT,'document_snippets':snippets}
 print(json.dumps(report,ensure_ascii=False,indent=2))
 return report
if __name__=='__main__':main()
