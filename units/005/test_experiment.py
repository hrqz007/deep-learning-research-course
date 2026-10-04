"""Boundary tests and an independent rational-arithmetic oracle."""
import copy, json, tempfile
from pathlib import Path
from fractions import Fraction
import numpy as np
import experiment as e

PASSED=[]
def check(name, fn):
    fn(); PASSED.append(name)
def rejects(fn, text=''):
    try: fn()
    except ValueError as err:
        if text: assert text in str(err), str(err)
    else: raise AssertionError('expected ValueError')

raw=e.read_csv(e.ROOT/'data/records_raw.csv')
rows,removed=e.clean_records(raw)
labels=e.read_csv(e.ROOT/'data/labels.csv')
joined=e.join_labels(rows,labels)
lists=json.loads((e.ROOT/'data/entity_split.json').read_text())
parts=e.split_records(joined,lists)
X=e.feature_matrix(parts['train']); state=e.fit_preprocessor(X)

def duplicates():
    assert len(rows)==12 and removed==['A02']
    assert {r['record_id'] for r in rows if r['device_id']=='A'}=={'A01','A02'}
    bad=copy.deepcopy(raw); bad[-1]['knob1']='99'
    rejects(lambda:e.clean_records(bad),'conflicting')
    bad=copy.deepcopy(rows); extra=bad[0].copy(); extra['record_id']='NEW'; bad.append(extra)
    rejects(lambda:e.clean_records(bad),'device/time')
    # Equal feature values on a different device are still separate observations.
    extra=rows[0].copy();extra.update(record_id='Z01',device_id='Z')
    assert len(e.clean_records(rows+[extra])[0])==13
check('exact duplicate vs legitimate repeated/equal-feature observations',duplicates)

def schema():
    rejects(lambda:e.clean_records([]),'empty')
    for column,value in [('knob1','NaN'),('knob2','inf'),('knob1',''),('knob2','broken'),('device_id','')]:
        bad=copy.deepcopy(rows);bad[0][column]=value
        rejects(lambda:e.clean_records(bad))
    bad=copy.deepcopy(rows);bad[0]['extra']='1';rejects(lambda:e.clean_records(bad),'columns')
check('schema, numeric parsing and finite-value failures',schema)

def label_join():
    assert joined[0]['target']==2 and joined[-1]['target']==47
    assert float(labels[0]['target'])==47 # Positional joining would be wrong.
    rejects(lambda:e.join_labels(rows,labels[:-1]),'key set')
    rejects(lambda:e.join_labels(rows,labels+[labels[0]]),'duplicate')
    bad=copy.deepcopy(labels);bad[0]['target']='NaN';rejects(lambda:e.join_labels(rows,bad),'non-finite')
check('identity join handles reversed labels and rejects missing or duplicate keys',label_join)

def splits():
    assert [len(parts[n]) for n in ['train','validation','test']]==[6,2,4]
    bad=copy.deepcopy(lists);bad['test'][0]='A01';rejects(lambda:e.split_records(joined,bad),'appears')
    bad=copy.deepcopy(lists);bad['test'][0]='Z01';rejects(lambda:e.split_records(joined,bad),'cover')
    bad=copy.deepcopy(lists);bad['train'][1],bad['validation'][0]=bad['validation'][0],bad['train'][1]
    rejects(lambda:e.split_records(joined,bad),'entity overlap')
    assert len(e.split_records(joined,bad,require_new_entities=False)['train'])==6
check('fixed lists exhaust rows without overlap and enforce target-specific entity isolation',splits)

def allowed():
    rejects(lambda:e.feature_matrix(rows,('knob1','post_reading')),'allowlist')
    rejects(lambda:e.feature_matrix(rows,('knob2','knob1')),'allowlist')
check('future field and silent feature-order changes rejected',allowed)

def arithmetic():
    np.testing.assert_allclose(state['mean'],[3.5,38]);np.testing.assert_allclose(state['span'],[5,50])
    filled,scaled=e.transform(X,state)
    np.testing.assert_allclose(filled[1],[2,38]);np.testing.assert_allclose(scaled[1],[.2,.56])
    v=e.feature_matrix(parts['validation']); vf,vs=e.transform(v,state)
    np.testing.assert_allclose(vs,[[1.2,1.2],[1.4,.56]])
    assert e.describe(X)['knob2']['missing_fraction']==1/6
    assert e.fit_preprocessor(e.feature_matrix(joined))['mean'][1]==68
    np.testing.assert_array_equal(np.histogram(X[:,0],bins=[1,3,5,7])[0],[2,2,2])
check('hand-calculated means, missing fraction, minmax scales and histogram counts',arithmetic)

def failed_preprocess():
    for bad in [np.ones((2,3)),np.ones(2),np.empty((0,2)),[[1,np.inf]]]:
        rejects(lambda:e.fit_preprocessor(bad))
    rejects(lambda:e.fit_preprocessor([[1,np.nan],[2,np.nan]]),'all-missing')
    rejects(lambda:e.fit_preprocessor([[1,2],[1,3]]),'constant')
    # A missing validation column is valid because TRAIN supplies the means.
    f,z=e.transform([[np.nan,np.nan]],state);np.testing.assert_allclose(f,[[3.5,38]])
check('empty, wrong shape, infinite, all-missing and constant TRAIN failures',failed_preprocess)

def invariant():
    original={k:v.copy() for k,v in state.items()}
    for val in [7,70000,-1000]:
        e.transform([[val,np.nan]],state)
    for k in state:np.testing.assert_array_equal(state[k],original[k])
    # Change every held-out input, then rebuild the split and refit only TRAIN.
    changed=copy.deepcopy(joined)
    for row in changed:
        if row['record_id'] not in lists['train']:row['knob1']='99999';row['knob2']='88888'
    again=e.fit_preprocessor(e.feature_matrix(e.split_records(changed,lists)['train']))
    for k in state:np.testing.assert_array_equal(again[k],state[k])
check('held-out changes cannot alter fitted training state',invariant)

def oracle():
    fixtures=0
    for n in (2,3,5,11):
        for start in range(-4,6):
            values=[[start+i,10*(start+i)+3] for i in range(n)]
            values[-1][1]=None if n>2 else values[-1][1]
            arr=np.array([[np.nan if x is None else x for x in row] for row in values],dtype=float)
            actual=e.fit_preprocessor(arr)
            means=[];lows=[];highs=[]
            for j in range(2):
                obs=[Fraction(row[j]) for row in values if row[j] is not None]
                mean=sum(obs)/len(obs); filled=[mean if row[j] is None else Fraction(row[j]) for row in values]
                means.append(mean);lows.append(min(filled));highs.append(max(filled))
            expected=[[float(((means[j] if row[j] is None else Fraction(row[j]))-lows[j])/(highs[j]-lows[j])) for j in range(2)] for row in values]
            np.testing.assert_allclose(actual['mean'],[float(v) for v in means],atol=1e-12)
            np.testing.assert_allclose(e.transform(arr,actual)[1],expected,atol=1e-12)
            fixtures+=1
    assert fixtures==40
    m=e.memorization_demo();assert m['same_entity_row_holdout_mae']==0
    assert abs(m['unseen_entity_holdout_mae']-float(Fraction(71,3)))<1e-12
check('40 independent Fraction fixtures and exact memorization 71/3 oracle',oracle)

if __name__=='__main__':
    print(json.dumps({'status':'passed','test_groups':PASSED,'independent_fraction_fixtures':40},ensure_ascii=False,indent=2))
