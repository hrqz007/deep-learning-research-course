"""Exact finite classification risk and generalization. Python standard library only."""
from __future__ import annotations
import argparse,csv,io,json,math,platform,random
from fractions import Fraction as F
from pathlib import Path
from itertools import product
HERE=Path(__file__).resolve().parent
CLASSES=('constant','threshold','lookup')


def integer(x,name,low,high):
    if type(x) is not int or not low<=x<=high:raise ValueError(f'{name}: integer in [{low},{high}] required')
    return x


def model(values):
    values=tuple(values)
    if len(values)!=8:raise ValueError('model must contain eight predictions')
    for y in values:integer(y,'prediction',0,1)
    return values


def observations(rows):
    rows=list(rows)
    if not 1<=len(rows)<=4096:raise ValueError('need 1 to 4096 observations')
    result=[]
    for row in rows:
        if not isinstance(row,(list,tuple)) or len(row)!=2:raise ValueError('each observation needs x,y')
        x,y=row;integer(x,'x',0,7);integer(y,'y',0,1);result.append((x,y))
    return result


def population(q=(1,1,1,1,9,9,9,9),weights=(1,)*8):
    q,weights=tuple(q),tuple(weights)
    if len(q)!=8 or len(weights)!=8:raise ValueError('eight q numerators and eight masses required')
    for x in q:integer(x,'q numerator over 10',0,10)
    for w in weights:integer(w,'x mass',0,10**6)
    if sum(weights)==0:raise ValueError('positive total mass required')
    return q,weights


def risk(predictions,q=(1,1,1,1,9,9,9,9),weights=(1,)*8):
    predictions=model(predictions);q,weights=population(q,weights)
    numerator=sum(w*(qi if h==0 else 10-qi) for w,qi,h in zip(weights,q,predictions))
    return F(numerator,10*sum(weights))


def empirical_risk(predictions,rows):
    predictions=model(predictions);rows=observations(rows)
    return F(sum(predictions[x]!=y for x,y in rows),len(rows))


def candidates(kind):
    if kind=='constant':return [(0,)*8,(1,)*8]
    if kind=='threshold':return [tuple(int(x>=t) for x in range(8)) for t in range(8,-1,-1)]
    if kind=='lookup':return list(product((0,1),repeat=8))
    raise ValueError('class must be constant, threshold, or lookup')


def fit(kind,rows):
    rows=observations(rows)
    if kind not in CLASSES:raise ValueError('unknown class')
    if kind=='lookup':
        zeros=[0]*8;ones=[0]*8
        for x,y in rows:(ones if y else zeros)[x]+=1
        return tuple(int(o>z) for o,z in zip(ones,zeros))
    return min(candidates(kind),key=lambda h:(sum(h[x]!=y for x,y in rows),h))


def load_training(path=HERE/'data/hand_training.csv'):
    with Path(path).open(newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=['sample_id','x','y']:raise ValueError('exact unique sample_id,x,y header required')
        rows=list(reader)
    if not rows or any(set(r)!={'sample_id','x','y'} for r in rows):raise ValueError('malformed training CSV')
    if [r['sample_id'] for r in rows]!=[str(i) for i in range(1,len(rows)+1)]:raise ValueError('consecutive unique sample IDs required')
    if any(r['x'] not in [str(i) for i in range(8)] or r['y'] not in ('0','1') for r in rows):raise ValueError('exact integer category strings required')
    return observations([(int(r['x']),int(r['y'])) for r in rows])


def sample(rng,n,q=(1,1,1,1,9,9,9,9)):
    integer(n,'n',1,4096);q,_=population(q)
    return [(x,int(rng.randrange(10)<q[x])) for x in [rng.randrange(8) for _ in range(n)]]


def checked_config(c):
    if not isinstance(c,dict) or set(c)!={'seed','repeats','sample_sizes','validation_size','test_size'}:raise ValueError('incorrect config fields')
    integer(c['seed'],'seed',0,2**32-1);integer(c['repeats'],'repeats',2,2000)
    for field in ('validation_size','test_size'):integer(c[field],field,2,4096)
    if not isinstance(c['sample_sizes'],list) or not 1<=len(c['sample_sizes'])<=8:raise ValueError('sample_sizes requires 1 to 8 entries')
    for n in c['sample_sizes']:integer(n,'sample size',1,4096)
    if len(set(c['sample_sizes']))!=len(c['sample_sizes']):raise ValueError('sample_sizes must be unique')
    if c['repeats']*sum(n+c['validation_size']+c['test_size'] for n in c['sample_sizes'])>3_000_000:raise ValueError('teaching CPU budget exceeded')
    return json.loads(json.dumps(c))


def simulate(config):
    c=checked_config(config);rng=random.Random(c['seed']);trials=[];summary=[]
    for n in c['sample_sizes']:
        rows_by_kind={kind:[] for kind in (*CLASSES,'validation_selected')}
        for repeat in range(c['repeats']):
            train=sample(rng,n);valid=sample(rng,c['validation_size']);test=sample(rng,c['test_size'])
            models={kind:fit(kind,train) for kind in CLASSES}
            # Selection only reads validation outcomes; no population risk or test labels enter.
            selected=min(CLASSES,key=lambda k:(sum(models[k][x]!=y for x,y in valid),CLASSES.index(k)))
            for kind in (*CLASSES,'validation_selected'):
                chosen=selected if kind=='validation_selected' else kind;h=models[chosen]
                row={'n':n,'repeat':repeat,'class':kind,'selected_class':chosen,
                     'train_risk':float(empirical_risk(h,train)),'validation_risk':float(empirical_risk(h,valid)),
                     'test_risk':float(empirical_risk(h,test)),'population_risk':float(risk(h))}
                trials.append(row);rows_by_kind[kind].append(row)
        for kind,rs in rows_by_kind.items():
            s={'n':n,'class':kind,'repeats':len(rs)}
            for key in ('train_risk','validation_risk','test_risk','population_risk'):
                vals=[r[key] for r in rs];avg=sum(vals)/len(vals)
                s['mean_'+key]=avg
                if key=='population_risk':s['mcse_population_mean']=math.sqrt(sum((v-avg)**2 for v in vals)/(len(vals)-1)/len(vals))
            summary.append(s)
    return trials,summary


def fixed_results(rows):
    rows=observations(rows);best=tuple(int(x>=4) for x in range(8));records={}
    for kind in CLASSES:
        h=fit(kind,rows);cs=candidates(kind);approx=min(risk(g) for g in cs)-F(1,10)
        dev=max(abs(empirical_risk(g,rows)-risk(g)) for g in cs)
        records[kind]={'predictions':list(h),'train_risk':str(empirical_risk(h,rows)),
            'population_risk':str(risk(h)),'class_best_risk':str(approx+F(1,10)),
            'approximation_error':str(approx),'estimation_error':str(risk(h)-approx-F(1,10)),
            'empirical_optimization_error':'0','uniform_deviation':str(dev),
            'excess_bound':str(approx+2*dev)}
    records['empirically_suboptimal_but_better_population']={'threshold4_train':str(empirical_risk(best,rows)),
        'threshold4_population':str(risk(best)),'population_change_from_erm':str(risk(best)-risk(fit('threshold',rows)))}
    records['shift']={'old_bayes_source':str(risk(best)),'old_bayes_reversed_conditional':str(risk(best,(9,9,9,9,1,1,1,1))),
        'learned_threshold_changed_x_masses':str(risk(fit('threshold',rows),weights=(1,3,3,3,1,1,1,1)))}
    records['squared_loss_at_x4']={'conditional_mean':'9/10','irreducible_noise':'9/100',
        'one_label_estimator_bias_squared':'0','one_label_estimator_variance':'9/100','one_label_total_risk':'9/50',
        'constant_half_bias_squared':'4/25','constant_half_variance':'0','constant_half_total_risk':'1/4'}
    return records


TEACHING_CONFIG={'seed':20261004,'repeats':800,'sample_sizes':[8,32,128],'validation_size':32,'test_size':256}
TEACHING_ROWS=[(0,0),(1,1),(4,1),(5,0)]


def require_teaching_inputs():
    config=json.loads((HERE/'data/config.json').read_text())
    rows=load_training(HERE/'data/hand_training.csv')
    previous=json.loads((HERE/'outputs/results.json').read_text())
    if config != TEACHING_CONFIG or rows != TEACHING_ROWS or previous.get('config') != TEACHING_CONFIG or previous.get('fixed') != fixed_results(TEACHING_ROWS):
        raise ValueError('Fixed teaching figures and Notebook require original config, training rows, and matching outputs; custom runs must be saved separately')
    return True


def csv_text(rows):
    buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows);return buf.getvalue()


def run(output=HERE/'outputs',config_path=HERE/'data/config.json',training_path=HERE/'data/hand_training.csv'):
    config=checked_config(json.loads(Path(config_path).read_text()));rows=load_training(training_path)
    records=fixed_results(rows);trials,summary=simulate(config)
    env={'python':platform.python_version(),'core_dependencies':'Python standard library only','seed':config['seed']}
    payload={'results.json':json.dumps({'fixed':records,'config':config},ensure_ascii=False,indent=2,allow_nan=False)+'\n',
             'trials.csv':csv_text(trials),'summary.csv':csv_text(summary),
             'environment.json':json.dumps(env,ensure_ascii=False,indent=2,allow_nan=False)+'\n'}
    for row in trials+summary:
        if any(isinstance(v,float) and not math.isfinite(v) for v in row.values()):raise ValueError('nonfinite results')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,value in payload.items():(output/name).write_text(value)
    return records,summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'outputs');p.add_argument('--config',type=Path,default=HERE/'data/config.json');p.add_argument('--training',type=Path,default=HERE/'data/hand_training.csv');args=p.parse_args()
    records,summary=run(args.output,args.config,args.training)
    print(json.dumps(records,ensure_ascii=False,indent=2));print('Summary rows:',len(summary))
