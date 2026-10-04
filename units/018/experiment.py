"""Finite expectations and sampling error. Core experiment: Python standard library.

Synthetic equal-probability atom IDs. No real training or population claim.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import math
from pathlib import Path
import platform
import random
from collections import Counter
from fractions import Fraction

HERE = Path(__file__).resolve().parent


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return value


def exact_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, Fraction, str)):
        raise ValueError('exact numbers must be int, Fraction, or rational string')
    try:
        q = Fraction(value)
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        raise ValueError('invalid exact number') from error
    if abs(q) > 10**12 or q.denominator > 10**12:
        raise ValueError('exact number exceeds teaching interface bounds')
    return q


def finite_moments(values, counts):
    """Return exact (mean, second moment, variance) for nonnegative integer masses."""
    values = [exact_number(v) for v in values]
    counts = list(counts)
    if not values or len(values) != len(counts) or len(values) > 1000:
        raise ValueError('values and counts need equal nonzero length, at most 1000')
    for c in counts:
        integer(c, 'count', 0, 10**9)
    total = sum(counts)
    if total == 0:
        raise ValueError('total mass must be positive')
    mu = sum((x*c for x,c in zip(values,counts)), Fraction()) / total
    second = sum((x*x*c for x,c in zip(values,counts)), Fraction()) / total
    variance = sum(((x-mu)**2*c for x,c in zip(values,counts)), Fraction()) / total
    assert variance == second - mu*mu
    return mu, second, variance


def covariance(xs, ys, counts):
    xs,ys,counts = list(xs),list(ys),list(counts)
    mx,_,_ = finite_moments(xs,counts)
    my,_,_ = finite_moments(ys,counts)
    if len(xs) != len(ys):
        raise ValueError('paired coordinates must have equal lengths')
    return sum(((exact_number(x)-mx)*(exact_number(y)-my)*c
                for x,y,c in zip(xs,ys,counts)),Fraction()) / sum(counts)


def sample_statistics(values):
    """Exact mean, n-1 sample variance, and its mean-variance estimate s²/n.

    The arithmetic is defined for any data; unbiasedness needs IID assumptions.
    """
    values = [exact_number(v) for v in values]
    n = len(values)
    if not 2 <= n <= 100000:
        raise ValueError('sample needs 2 to 100000 observations')
    mean = sum(values,Fraction()) / n
    s2 = sum(((x-mean)**2 for x in values),Fraction()) / (n-1)
    return mean,s2,s2/n


def sample_mean_variance(variance, n, copies=1):
    """Exact true variance with n/copies independent, equal-sized copied groups."""
    variance=exact_number(variance)
    if variance < 0:
        raise ValueError('variance must be nonnegative')
    integer(n,'n',1,100000)
    integer(copies,'copies',1,n)
    if n % copies:
        raise ValueError('n must be divisible by copies')
    return variance * copies / n


def exact_mean_distribution(atoms, n):
    """Convolve integer sums; atom IDs are equally likely, repeated values allowed."""
    atoms=list(atoms)
    if not 1 <= len(atoms) <= 20:
        raise ValueError('need 1 to 20 atoms')
    for x in atoms:integer(x,'atom',-100,100)
    integer(n,'n',1,64)
    counts=Counter({0:1})
    for _ in range(n):
        nxt=Counter()
        for total,count in counts.items():
            for x in atoms:nxt[total+x]+=count
        counts=nxt
    denominator=len(atoms)**n
    return [(Fraction(s,n),Fraction(c,denominator)) for s,c in sorted(counts.items())]


def checked_config(config):
    if not isinstance(config,dict) or set(config) != {'seed','repeats','sample_sizes','copies','atoms'}:
        raise ValueError('config must have exactly seed repeats sample_sizes copies atoms')
    integer(config['seed'],'seed',0,2**32-1)
    integer(config['repeats'],'repeats',2,20000)
    integer(config['copies'],'copies',2,100)
    atoms=config['atoms']
    if not isinstance(atoms,list) or not 2 <= len(atoms) <= 20:
        raise ValueError('atoms must be a list of 2 to 20 integer atom values')
    for x in atoms:integer(x,'atom',-100,100)
    ns=config['sample_sizes']
    if not isinstance(ns,list) or not 1 <= len(ns) <= 10 or len(set(map(str,ns))) != len(ns):
        raise ValueError('sample_sizes must be a nonempty unique list of at most 10')
    for n in ns:
        integer(n,'sample size',2,4096)
        if n % config['copies'] or n // config['copies'] < 2:
            raise ValueError('each n needs at least two complete copied groups')
    if config['repeats']*sum(ns) > 5_000_000:
        raise ValueError('simulation exceeds teaching CPU budget')
    return json.loads(json.dumps(config))


def fast_stats(values):
    """Bounded integer simulation rows, centered-sum formula with no huge numbers."""
    n=len(values);s=sum(values);ss=sum(x*x for x in values)
    mean=s/n
    variance=(ss-s*s/n)/(n-1)
    return mean,variance


def simulate(config):
    config=checked_config(config)
    rng=random.Random(config['seed']);atoms=config['atoms'];trials=[];summary=[]
    mu,_,variance=finite_moments(atoms,[1]*len(atoms))
    for n in config['sample_sizes']:
        for mode,copies in [('iid',1),('copied_groups',config['copies'])]:
            group_count=n//copies;rows=[]
            for r in range(config['repeats']):
                independent=[atoms[rng.randrange(len(atoms))] for _ in range(group_count)]
                mean,group_s2=fast_stats(independent)
                # Expanded row centered sum is copies times the group centered sum.
                row_s2=copies*(group_count-1)*group_s2/(n-1)
                naive_se=math.sqrt(row_s2/n)
                group_se=math.sqrt(group_s2/group_count)
                row={'mode':mode,'n':n,'repeat':r,'independent_groups':group_count,
                     'sample_mean':mean,'naive_se':naive_se,'unit_correct_se':group_se}
                rows.append(row);trials.append(row)
            means=[row['sample_mean'] for row in rows]
            avg=sum(means)/len(means)
            empirical_var=sum((x-avg)**2 for x in means)/(len(means)-1)
            true_var=sample_mean_variance(variance,n,copies)
            summary.append({'mode':mode,'n':n,'independent_groups':group_count,
                'replicates':len(means),'mean_of_means':avg,'observed_bias':avg-float(mu),
                'empirical_sd':math.sqrt(empirical_var),'true_se':math.sqrt(float(true_var)),
                'rms_naive_se':math.sqrt(sum(row['naive_se']**2 for row in rows)/len(rows)),
                'rms_unit_correct_se':math.sqrt(sum(row['unit_correct_se']**2 for row in rows)/len(rows)),
                'mcse_of_mean_of_means':math.sqrt(empirical_var/len(means))})
    return trials,summary


def fixed_results():
    mu,second,var=finite_moments([0,2,6],[2,1,1])
    sample=[0,0,2,6,0,2,0,6]
    mean,s2,se2=sample_statistics(sample)
    copied=[x for x in sample for _ in range(8)]
    _,copied_s2,naive_se2=sample_statistics(copied)
    result={'population':{'mean':str(mu),'second_moment':str(second),'variance':str(var)},
       'fixed_eight_groups':{'values':sample,'mean':str(mean),'sample_variance':str(s2),
            'estimated_mean_variance':str(se2),'estimated_se':math.sqrt(float(se2))},
       'same_groups_copied_eight_times':{'rows':64,'mean':str(mean),'row_sample_variance':str(copied_s2),
            'naive_estimated_mean_variance':str(naive_se2),'naive_se':math.sqrt(float(naive_se2)),
            'unit_correct_se':math.sqrt(float(se2))},
       'true_mean_variance_n64':{'iid':str(sample_mean_variance(var,64)),
            'copied_eight':str(sample_mean_variance(var,64,8))},
       'uncorrelated_dependent':{'covariance':str(covariance([-1,0,1],[1,0,1],[1,1,1]))},
       'rare_event':{'p':'1/1000','n':100,'prob_all_zero':float(Fraction(999,1000)**100),
            'true_se':math.sqrt(.001*.999/100),'observed_se_if_all_zero':0.0}}
    return result


TEACHING_CONFIG = {'seed':20261004,'repeats':4000,'sample_sizes':[16,32,64,128],
                   'copies':8,'atoms':[0,0,2,6]}


def require_teaching_config(config_path=HERE/'data/config.json',results_path=HERE/'outputs/results.json'):
    """Fixed pedagogical figures/Notebook only: refuse mixed/custom experiment inputs."""
    config=json.loads(Path(config_path).read_text())
    previous=json.loads(Path(results_path).read_text())
    if config != TEACHING_CONFIG or previous.get('config') != TEACHING_CONFIG:
        raise ValueError('Fixed teaching figures and Notebook require the original default config and outputs; save custom simulations separately')
    return True


def csv_text(rows):
    buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows);return buf.getvalue()


def run(output=HERE/'outputs',config_path=HERE/'data/config.json'):
    config=checked_config(json.loads(Path(config_path).read_text()))
    trials,summary=simulate(config)
    records=fixed_results();records['config']=config
    env={'python':platform.python_version(),'implementation':platform.python_implementation(),
         'core_dependencies':'Python standard library only','seed':config['seed']}
    # Finish all validation and serialization before any output directory or file mutation.
    payload={'results.json':json.dumps(records,ensure_ascii=False,indent=2,allow_nan=False)+'\n',
             'sampling_trials.csv':csv_text(trials),'sampling_summary.csv':csv_text(summary),
             'environment.json':json.dumps(env,ensure_ascii=False,indent=2,allow_nan=False)+'\n'}
    for row in trials+summary:
        if any(isinstance(v,float) and not math.isfinite(v) for v in row.values()):
            raise ValueError('nonfinite output rejected before writing')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,text in payload.items():(output/name).write_text(text)
    return records,summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'outputs')
    parser.add_argument('--config',type=Path,default=HERE/'data/config.json');args=parser.parse_args()
    result,summary=run(args.output,args.config)
    print(json.dumps(result,ensure_ascii=False,indent=2));print('Summary rows:',len(summary))
