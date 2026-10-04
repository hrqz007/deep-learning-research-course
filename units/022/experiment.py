"""Offline, bounded paired-comparison teaching cases. All data are synthetic."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import csv, io, json, math, random, sys
HERE=Path(__file__).resolve().parent
DEFAULT_CONFIG={'seed':20261004,'bootstrap_repeats':4000,'selection_repeats':2000}
DEFAULT_ROWS=[(i,a,b) for i,(a,b) in enumerate(zip([30,20,40,25,35,20,30,40],[28,16,40,19,37,16,28,40]))]
PLAN={'comparison':['A','B'],'metric':'error_percent','effect':'A_minus_B_percentage_points','paired_unit':'training_seed_on_fixed_data','seed_ids':list(range(8)),'selection':'none_all_eight_pairs_reported','intervals':['paired_t7_rounded','percentile_bootstrap'],'bootstrap_unit':'whole_seed_pair','practical_margin_points':3}
T7=2.365 # NIST table, df=7, two-sided 95%; rounded, not exact critical value

def integer(x,lo,hi,name):
 if type(x) is not int or not lo<=x<=hi:raise ValueError(f'{name}: integer in [{lo},{hi}] required')
 return x

def validate_values(values,lo=-100,hi=100,min_n=2,max_n=1000):
 if not isinstance(values,(list,tuple)) or not min_n<=len(values)<=max_n:raise ValueError('invalid number of values')
 for x in values:integer(x,lo,hi,'value')
 return tuple(values)

def moments(values):
 x=validate_values(values);n=len(x);m=F(sum(x),n);v=sum((F(y)-m)**2 for y in x)/(n-1)
 return m,v

def paired_summary(a,b):
 a=validate_values(a,0,100);b=validate_values(b,0,100)
 if len(a)!=len(b):raise ValueError('pairs must have equal lengths')
 d=tuple(x-y for x,y in zip(a,b));m,v=moments(d);ma,va=moments(a);mb,vb=moments(b)
 cov=sum((F(x)-ma)*(F(y)-mb) for x,y in zip(a,b))/(len(a)-1)
 return {'n':len(d),'differences':list(d),'mean':float(m),'sample_variance':float(v),'sample_covariance':float(cov),'paired_se':math.sqrt(float(v/len(d))),'unpaired_formula_se':math.sqrt(float((va+vb)/len(d))), 'exact_mean':str(m),'exact_variance':str(v),'variance_identity':str(va+vb-2*cov)}

def interval(values,critical):
 x=validate_values(values)
 if isinstance(critical,bool) or not isinstance(critical,(float,int)) or not math.isfinite(critical) or not 0<critical<=100:raise ValueError('invalid critical value')
 m,v=moments(x);half=critical*math.sqrt(float(v/len(x)))
 return [float(m)-half,float(m)+half]

def paired_t7(values):
 x=validate_values(values)
 if len(x)!=8:raise ValueError('table critical value only supports n=8')
 return interval(x,T7)

def convolution_counts(values):
 """All n**n ordered empirical bootstrap samples, compressed by integer sum."""
 x=validate_values(values,min_n=2,max_n=10);single=Counter(x);counts=Counter({0:1})
 for _ in x:
  nxt=Counter()
  for a,c in counts.items():
   for b,k in single.items():nxt[a+b]+=c*k
  counts=nxt
 return dict(sorted(counts.items()))

def quantile_counts(counts,q,scale=1):
 if not isinstance(q,F) or not 0<=q<=1:raise ValueError('q must be Fraction in [0,1]')
 integer(scale,1,100000,'scale')
 if not isinstance(counts,dict) or not counts:raise ValueError('nonempty count mapping required')
 for x,c in counts.items():integer(x,-100000,100000,'sum');integer(c,1,10**100,'count')
 total=sum(counts.values());acc=0
 for x,c in sorted(counts.items()):
  acc+=c
  if acc*q.denominator>=q.numerator*total:return F(x,scale)
 raise ArithmeticError('quantile calculation failed')

def exact_bootstrap(values):
 x=validate_values(values,min_n=2,max_n=10);counts=convolution_counts(x);total=sum(counts.values());n=len(x)
 mu=sum(F(s,n)*c for s,c in counts.items())/total
 var=sum((F(s,n)-mu)**2*c for s,c in counts.items())/total
 return {'ordered_samples':total,'mean':str(mu),'variance':str(var),'percentile_95':[str(quantile_counts(counts,F(1,40),n)),str(quantile_counts(counts,F(39,40),n))],'counts':counts}

def bootstrap_means(values,repeats,seed):
 x=validate_values(values,max_n=50);integer(repeats,20,20000,'bootstrap_repeats');integer(seed,0,2**32-1,'seed');rng=random.Random(seed)
 return [F(sum(x[rng.randrange(len(x))] for _ in x),len(x)) for _ in range(repeats)]

def empirical_quantile(values,q):
 if not isinstance(values,(tuple,list)) or not values or any(not isinstance(x,F) for x in values):raise ValueError('nonempty Fraction sample required')
 if not isinstance(q,F) or not 0<=q<=1:raise ValueError('invalid quantile')
 vals=sorted(values);i=max(0,(q.numerator*len(vals)+q.denominator-1)//q.denominator-1);return vals[i]

def rare_coverage(n=8,p=F(1,100)):
 """Exact coverage of percentile bootstrap for iid 20*Bernoulli(p)."""
 integer(n,2,10,'n')
 if not isinstance(p,F) or not 0<p<1:raise ValueError('p must be Fraction strictly inside (0,1)')
 mu=20*p;rows=[];coverage=F(0)
 for k in range(n+1):
  mass=math.comb(n,k)*p**k*(1-p)**(n-k)
  # Counts proportional to Binomial(n,k/n): empirical bootstrap binary resamples.
  counts={20*j:math.comb(n,j)*k**j*(n-k)**(n-j) for j in range(n+1)};counts={x:c for x,c in counts.items() if c}
  low=quantile_counts(counts,F(1,40),n);high=quantile_counts(counts,F(39,40),n);hit=low<=mu<=high
  coverage+=mass*hit;rows.append({'positive_count':k,'sample_probability':str(mass),'lower':str(low),'upper':str(high),'covers_true_mean':int(hit)})
 return coverage,rows

def null_selection(repeats,seed):
 integer(repeats,20,5000,'selection_repeats');integer(seed,0,2**32-1,'seed');rng=random.Random(seed);rows=[]
 for m in (1,5,20):
  for trial in range(repeats):
   validation=[rng.choice((-1,1)) for _ in range(m)];winner=max(range(m),key=lambda i:validation[i]);test=rng.choice((-1,1))
   rows.append({'candidates':m,'trial':trial,'winner':winner,'selected_validation_noise':validation[winner],'independent_test_noise':test})
 return rows

def read_rows(path):
 with Path(path).open(newline='',encoding='utf-8') as f:
  reader=csv.DictReader(f)
  if reader.fieldnames!=['seed_id','error_A','error_B']:raise ValueError('CSV header must match exactly')
  rows=[]
  for row in reader:
   if set(row)!=set(reader.fieldnames) or any(v is None or not v.isdecimal() for v in row.values()):raise ValueError('complete nonnegative integer CSV fields required')
   rows.append(tuple(int(row[x]) for x in reader.fieldnames))
 if len(rows)!=8 or [r[0] for r in rows]!=list(range(8)):raise ValueError('exactly seed IDs 0..7 in order required')
 for _,a,b in rows:integer(a,0,100,'error_A');integer(b,0,100,'error_B')
 return rows

def read_json(path):
 def unique(pairs):
  d={}
  for k,v in pairs:
   if k in d:raise ValueError('duplicate JSON key')
   d[k]=v
  return d
 return json.loads(Path(path).read_text(),object_pairs_hook=unique)

def validate_config(config):
 if type(config) is not dict or set(config)!=set(DEFAULT_CONFIG):raise ValueError('exact config keys required')
 integer(config['seed'],0,2**32-1,'seed');integer(config['bootstrap_repeats'],20,20000,'bootstrap_repeats');integer(config['selection_repeats'],20,5000,'selection_repeats')
 return config

def csv_text(rows):
 s=io.StringIO(newline='');writer=csv.DictWriter(s,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows);return s.getvalue()

def require_teaching_inputs():
 if read_json(HERE/'data/config.json')!=DEFAULT_CONFIG or read_json(HERE/'data/plan.json')!=PLAN or read_rows(HERE/'data/seed_scores.csv')!=DEFAULT_ROWS:raise ValueError('fixed teaching figures require original config, plan and seed rows')
 r=read_json(HERE/'outputs/results.json')
 if r.get('config')!=DEFAULT_CONFIG or r.get('plan')!=PLAN or r.get('input_rows')!=[list(row) for row in DEFAULT_ROWS]:raise ValueError('saved results do not match fixed teaching inputs')

def run(output,config_path=None,rows_path=None,plan_path=None):
 config=validate_config(read_json(config_path or HERE/'data/config.json'));plan=read_json(plan_path or HERE/'data/plan.json')
 if plan!=PLAN:raise ValueError('this teaching implementation supports only the declared comparison plan')
 rows=read_rows(rows_path or HERE/'data/seed_scores.csv');a=[r[1] for r in rows];b=[r[2] for r in rows];summary=paired_summary(a,b);d=summary['differences']
 boot=bootstrap_means(d,config['bootstrap_repeats'],config['seed']);exact=exact_bootstrap(d);coverage,rare=rare_coverage();selection=null_selection(config['selection_repeats'],config['seed']+1 if config['seed']<2**32-1 else 0)
 selected=[]
 for m in (1,5,20):
  subset=[r for r in selection if r['candidates']==m]
  selected.append({'candidates':m,'mean_selected_validation':sum(r['selected_validation_noise'] for r in subset)/len(subset),'mean_independent_test':sum(r['independent_test_noise'] for r in subset)/len(subset),'theoretical_selected_mean':str(1-F(2,2**m)),'familywise_05_independent':str(1-F(19,20)**m),'bonferroni_level':str(F(1,20*m))})
 result={'config':config,'plan':plan,'input_rows':rows,'paired':summary,'normal_approx_95':interval(d,1.96),'t7_rounded_95':paired_t7(d),'bootstrap_monte_carlo_95':[str(empirical_quantile(boot,F(1,40))),str(empirical_quantile(boot,F(39,40)))],'bootstrap_exact':{k:v for k,v in exact.items() if k!='counts'},'rare_bootstrap_exact_coverage':str(coverage),'rare_all_zero_probability':str(F(99,100)**8),'selection_summary':selected,'cluster_demo':{'four_cluster_means':[-2,0,2,4],'cluster_se_squared':'5/3','naive_32row_se_squared':'5/31'},'ablation_error':{'00':10,'10':8,'01':9,'11':4}}
 environment={'python':sys.version.split()[0],'core':'Python standard library only','data':'synthetic','seed':config['seed']}
 texts={'results.json':json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n','bootstrap.csv':csv_text([{'replicate':i,'mean_difference':str(x)} for i,x in enumerate(boot)]),'selection.csv':csv_text(selection),'rare_coverage.csv':csv_text(rare),'environment.json':json.dumps(environment,indent=2)+'\n'}
 # All validation, calculations and serialization precede any output mutation.
 output=Path(output)
 if output.is_symlink() or any((output/n).is_symlink() for n in texts):raise ValueError('output symlinks are not supported')
 if output.exists() and not output.is_dir():raise ValueError('output must be a directory')
 output.mkdir(parents=True,exist_ok=True)
 for name,text in texts.items():(output/name).write_text(text,encoding='utf-8')
 return result

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'outputs');p.add_argument('--config',type=Path);p.add_argument('--rows',type=Path);p.add_argument('--plan',type=Path);args=p.parse_args()
 r=run(args.output,args.config,args.rows,args.plan);print(json.dumps({'mean':r['paired']['mean'],'paired_se':r['paired']['paired_se'],'t7':r['t7_rounded_95'],'bootstrap_exact':r['bootstrap_exact'],'rare_coverage':r['rare_bootstrap_exact_coverage']},indent=2))
