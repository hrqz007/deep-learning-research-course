"""Exact finite probability, classification posterior, and causal toy models."""
from pathlib import Path
from fractions import Fraction
import argparse,csv,json,platform,sys
BASE=Path(__file__).resolve().parent
OUTCOMES={(0,0),(0,1),(1,0),(1,1)}


def bit(value):
    if type(value) is not int or value not in (0,1):raise ValueError('Labels must be integer 0 or 1, not bool')
    return value


def outcome(value):
    if not isinstance(value,(tuple,list)) or len(value)!=2:raise ValueError('Outcome must have two binary labels')
    return bit(value[0]),bit(value[1])


class FiniteTable:
    def __init__(self,counts):
        if not isinstance(counts,dict):raise TypeError('Counts must be a dictionary')
        clean={}
        for key,count in counts.items():
            pair=outcome(key)
            if type(count) is not int or count<0:raise ValueError('Counts must be nonnegative integers, not bool')
            clean[pair]=count
        if set(clean)!=OUTCOMES:raise ValueError('All four binary outcomes must be explicit')
        if sum(clean.values())<=0:raise ValueError('Population must be nonempty')
        self.counts=clean;self.total=sum(clean.values())

    def event(self,values):
        if not isinstance(values,(list,tuple,set,frozenset)):raise TypeError('An event must be a finite collection of outcomes')
        return {outcome(value) for value in values}

    def probability(self,event):
        e=self.event(event)
        return Fraction(sum(self.counts[key] for key in e),self.total)

    def conditional(self,event,given):
        a,b=self.event(event),self.event(given)
        denominator=self.probability(b)
        if denominator==0:raise ValueError('Conditional probability requires positive conditioning mass')
        return self.probability(a & b)/denominator

    def independent(self,event_a,event_b):
        a,b=self.event(event_a),self.event(event_b)
        return self.probability(a & b)==self.probability(a)*self.probability(b)


def load_table():
    with (BASE/'data/inspection_counts.csv').open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['defective','flagged','count']:raise ValueError('Unexpected CSV schema')
        counts={}
        for row in reader:
            if set(row) != {'defective','flagged','count'} or any(not isinstance(value,str) or not value.isdecimal() for value in row.values()):raise ValueError('CSV values must be three nonnegative integers')
            pair=outcome((int(row['defective']),int(row['flagged'])))
            if pair in counts:raise ValueError('Duplicate outcome row')
            counts[pair]=int(row['count'])
    return FiniteTable(counts)


def rational_probability(value):
    if isinstance(value,bool) or not isinstance(value,(int,str,Fraction)):raise TypeError('Use integer, fraction string, or Fraction; not binary float')
    try:p=Fraction(value)
    except (ValueError,ZeroDivisionError) as error:raise ValueError('Invalid rational probability') from error
    if p<0 or p>1:raise ValueError('Probability must lie in [0,1]')
    return p


def bayes_flag(prevalence,hit_rate,false_alarm):
    prior,hit,false=map(rational_probability,(prevalence,hit_rate,false_alarm))
    if prior==0 or prior==1:raise ValueError("Both class branches must have positive probability to define their observed conditional rates")
    positive_defect=prior*hit
    positive_good=(1-prior)*false
    mass=positive_defect+positive_good
    if mass==0:raise ValueError('Flag conditioning event has zero probability')
    return {'prior':prior,'joint_defect_flag':positive_defect,'joint_good_flag':positive_good,
            'flag_probability':mass,'posterior_defect_given_flag':positive_defect/mass}


def xor_world():
    rows=[{'x':x,'y':y,'z':(x+y)%2,'mass':Fraction(1,4)} for x in (0,1) for y in (0,1)]
    pairs={}
    for a,b in [('x','y'),('x','z'),('y','z')]:
        pairs[a+b]=all(sum(r['mass'] for r in rows if r[a]==i and r[b]==j)==Fraction(1,4) for i in (0,1) for j in (0,1))
    return {'rows':rows,'pairwise_independent':pairs,'triple_111':sum((r['mass'] for r in rows if r['x']==r['y']==r['z']==1),Fraction(0)),
            'product_of_three_marginals':Fraction(1,8)}


def causal_world(model,intervene_x=None):
    if model not in ('chain','common_cause'):raise ValueError('Unknown model')
    if intervene_x is not None:bit(intervene_x)
    rows=[]
    for u in (0,1):
        x=u if intervene_x is None else intervene_x
        y=x if model=='chain' else u
        rows.append({'u':u,'x':x,'y':y,'mass':Fraction(1,2)})
    return rows


def replacement_example():
    # Five distinct objects, two defective; enumerate every ordered pair.
    items=[('a',1),('b',1),('c',0),('d',0),('e',0)]
    result={}
    for replace in [True,False]:
        pairs=[(x,y) for x in items for y in items if replace or x[0]!=y[0]]
        first=[pair for pair in pairs if pair[0][1]==1]
        both=[pair for pair in first if pair[1][1]==1]
        result['with_replacement' if replace else 'without_replacement']={
            'ordered_pairs':len(pairs),'both_defective':Fraction(len(both),len(pairs)),
            'second_given_first_defective':Fraction(len(both),len(first))}
    return result


def encode(value):
    if isinstance(value,Fraction):return {'fraction':str(value),'decimal':float(value)}
    if isinstance(value,dict):return {str(k):encode(v) for k,v in value.items()}
    if isinstance(value,list):return [encode(v) for v in value]
    return value


def run(output_dir=None):
    table=load_table();D={(1,0),(1,1)};F={(0,1),(1,1)}
    good=OUTCOMES-D
    report={'population':table.total,'joint':{f'{a}{b}':table.probability({(a,b)}) for a,b in sorted(OUTCOMES)},
            'defect_probability':table.probability(D),'flag_probability':table.probability(F),
            'flag_given_defect':table.conditional(F,D),'flag_given_good':table.conditional(F,good),
            'defect_given_flag':table.conditional(D,F),'independent':table.independent(D,F),
            'bayes_reconstruction':bayes_flag(table.probability(D),table.conditional(F,D),table.conditional(F,good)),
            'replacement':replacement_example(),'xor':xor_world(),'causal_models':{}}
    for model in ('chain','common_cause'):
        observed=causal_world(model);do=causal_world(model,1)
        report['causal_models'][model]={'observed_rows':observed,'intervention_rows':do,
            'P_Y1_given_X1':sum(r['mass'] for r in observed if r['x']==r['y']==1)/sum(r['mass'] for r in observed if r['x']==1),
            'P_Y1_do_X1':sum(r['mass'] for r in do if r['y']==1)}
    scan=[bayes_flag(p,'9/10','1/10') for p in ['1/1000','1/100','1/50','1/10','1/2']]
    output=Path(output_dir) if output_dir is not None else BASE/'outputs';output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(encode(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (output/'prevalence_scan.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['prior','flag_probability','posterior_fraction','posterior_decimal']);writer.writeheader()
        for row in scan:writer.writerow({'prior':str(row['prior']),'flag_probability':str(row['flag_probability']),'posterior_fraction':str(row['posterior_defect_given_flag']),'posterior_decimal':float(row['posterior_defect_given_flag'])})
    (output/'environment.json').write_text(json.dumps({'python':sys.version.split()[0],'platform':platform.system(),'arithmetic':'exact Fraction; float only for display','network_used':False},indent=2)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    print(json.dumps(encode(run(args.output)),ensure_ascii=False,indent=2,allow_nan=False))
