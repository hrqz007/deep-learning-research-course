"""Unit 005: explicit information boundaries, not a trained neural network."""
from pathlib import Path
from datetime import datetime
import argparse, csv, hashlib, json, platform
import numpy as np

ROOT = Path(__file__).resolve().parent
FEATURES = ('knob1','knob2')
REQUIRED = {'record_id','device_id','predict_time','knob1','knob2','post_reading','post_available_time'}

def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def clean_records(raw):
    """Drop identical repeated record IDs; reject conflicts, never dedup by device."""
    if not raw:
        raise ValueError('empty records')
    kept, seen, business_keys, removed = [], {}, {}, []
    for row in raw:
        if set(row) != REQUIRED:
            raise ValueError('unexpected record columns')
        if not row['record_id'] or not row['device_id']:
            raise ValueError('empty identifier')
        for field in ('predict_time','post_available_time'):
            datetime.fromisoformat(row[field])
        for field in FEATURES + ('post_reading',):
            if row[field] == '' and field == 'knob2':
                continue
            try: value = float(row[field])
            except (TypeError, ValueError) as error:
                raise ValueError('non-numeric '+field) from error
            if not np.isfinite(value):
                raise ValueError('non-finite '+field)
        key = row['record_id']
        if key in seen:
            if row != seen[key]:
                raise ValueError('conflicting record_id: '+key)
            removed.append(key); continue
        business = (row['device_id'], row['predict_time'])
        if business in business_keys:
            raise ValueError('duplicate device/time observation requires review')
        seen[key] = row.copy(); business_keys[business] = key
        kept.append(row.copy())
    return kept, removed

def join_labels(rows, labels):
    """Join by identity, not CSV order; fail for absent, extra or repeated keys."""
    index = {}
    for row in labels:
        if set(row) != {'record_id','target','label_available_time'}:
            raise ValueError('unexpected label columns')
        key = row['record_id']
        if key in index: raise ValueError('duplicate label key')
        value = float(row['target'])
        if not np.isfinite(value): raise ValueError('non-finite target')
        datetime.fromisoformat(row['label_available_time'])
        index[key] = row
    if set(index) != {row['record_id'] for row in rows}:
        raise ValueError('label key set mismatch')
    return [{**row,'target':float(index[row['record_id']]['target']),
             'label_available_time':index[row['record_id']]['label_available_time']}
            for row in rows]

def split_records(rows, lists, require_new_entities=True):
    if set(lists) != {'train','validation','test'}:
        raise ValueError('expected train/validation/test lists')
    flat = [key for group in lists.values() for key in group]
    if any(not group for group in lists.values()):
        raise ValueError('empty partition')
    if len(flat) != len(set(flat)):
        raise ValueError('record appears in more than one list or repeats')
    index = {row['record_id']:row for row in rows}
    if len(index) != len(rows) or set(flat) != set(index):
        raise ValueError('split does not cover unique observations exactly')
    result = {name:[index[key] for key in keys] for name,keys in lists.items()}
    if require_new_entities:
        entities = [{row['device_id'] for row in group} for group in result.values()]
        if any(entities[i] & entities[j] for i in range(3) for j in range(i)):
            raise ValueError('entity overlap violates unseen-device target')
    return result

def feature_matrix(rows, columns=FEATURES):
    if tuple(columns) != FEATURES:
        raise ValueError('feature allowlist is exactly knob1, knob2')
    if not rows: raise ValueError('empty feature batch')
    return np.array([[float(row[c]) if row[c] != '' else np.nan for c in columns]
                     for row in rows], dtype=np.float64)

def validate_matrix(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 2 or X.shape[0] == 0:
        raise ValueError('expected nonempty (N,2) features')
    if np.isinf(X).any(): raise ValueError('infinity is not missing')
    return X

def fit_preprocessor(X_train):
    """Caller supplies TRAIN only; arrays alone cannot prove provenance."""
    X = validate_matrix(X_train)
    observed = ~np.isnan(X); count = observed.sum(axis=0)
    if (count == 0).any(): raise ValueError('all-missing training feature')
    mean = np.where(observed, X, 0).sum(axis=0) / count
    filled = np.where(observed, X, mean)
    low, high = filled.min(axis=0), filled.max(axis=0)
    span = high-low
    if (span == 0).any(): raise ValueError('constant training feature requires explicit policy')
    return {'mean':mean,'low':low,'high':high,'span':span,'observed_count':count}

def transform(X, state):
    X = validate_matrix(X)
    filled = np.where(np.isnan(X), state['mean'], X)
    return filled, (filled-state['low']) / state['span']

def describe(X):
    X = validate_matrix(X); observed = ~np.isnan(X)
    result = {}
    for j,name in enumerate(FEATURES):
        values = X[observed[:,j],j]
        if len(values)==0: raise ValueError('cannot describe an entirely missing column')
        result[name] = {'rows':len(X),'observed':len(values),'missing':int((~observed[:,j]).sum()),
                        'missing_fraction':float((~observed[:,j]).sum()/len(X)),
                        'sum':float(values.sum()),'mean':float(values.sum()/len(values)),
                        'min':float(values.min()),'max':float(values.max())}
    return result

def memorization_demo(path=ROOT/'data/memorization.csv'):
    rows=read_csv(path)
    def train_memory(train):
        # This fixture has constant per-device targets. General repeated labels need a policy.
        return {r['device_id']:float(r['target']) for r in train}
    def score(train, test):
        memory=train_memory(train)
        fallback=sum(float(r['target']) for r in train)/len(train)
        errors=[abs(memory.get(r['device_id'],fallback)-float(r['target'])) for r in test]
        return sum(errors)/len(errors)
    row_train=[r for r in rows if r['visit']=='1']; row_test=[r for r in rows if r['visit']=='2']
    group_train=[r for r in rows if r['device_id'] in 'ABC']; group_test=[r for r in rows if r['device_id'] in 'DEF']
    return {'same_entity_row_holdout_mae':score(row_train,row_test),
            'unseen_entity_holdout_mae':score(group_train,group_test),
            'row_split_entity_overlap':['A','B','C','D','E','F'],
            'unseen_entity_fallback':6.0,
            'note':'two different evaluation targets; tiny constructed counterexample, not a performance estimate'}

def run(output_dir):
    raw = read_csv(ROOT/'data/records_raw.csv'); clean, removed = clean_records(raw)
    rows = join_labels(clean,read_csv(ROOT/'data/labels.csv'))
    lists=json.loads((ROOT/'data/entity_split.json').read_text())
    parts=split_records(rows,lists)
    matrices={name:feature_matrix(group) for name,group in parts.items()}
    state=fit_preprocessor(matrices['train'])
    # Incorrect all-data fit is deliberately separate and never used for official arrays.
    leaked=fit_preprocessor(feature_matrix(rows))
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    ledger={}
    for name,X in matrices.items():
        filled, scaled=transform(X,state)
        ledger[name]={'ids':lists[name],'raw':[[None if np.isnan(v) else float(v) for v in row] for row in X],
                      'filled':filled.tolist(),'scaled':scaled.tolist(),'shape':list(scaled.shape)}
    state_json={k:v.tolist() for k,v in state.items()}
    report={'raw_rows':len(raw),'unique_rows':len(rows),'removed_exact_duplicate_ids':removed,
            'split_sizes':{k:len(v) for k,v in parts.items()},'train_description':describe(matrices['train']),
            'fit_record_ids':lists['train'],'feature_columns':list(FEATURES),
            'forbidden_post_reading_after_prediction':all(datetime.fromisoformat(r['post_available_time'])>datetime.fromisoformat(r['predict_time']) for r in rows),
            'wrong_all_data_knob2_mean':float(leaked['mean'][1]),'correct_train_knob2_mean':float(state['mean'][1]),
            'train_knob1_histogram':np.histogram(matrices['train'][:,0],bins=[1,3,5,7])[0].tolist(),
            'memory_counterexample':memorization_demo(),
            'test_targets_scored':False}
    for filename,value in [('preprocessor.json',state_json),('tensor_ledger.json',ledger),('audit.json',report),('split_lists.json',lists)]:
        (out/filename).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    env={'python':platform.python_version(),'numpy':np.__version__,'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'data').glob('*')) if p.suffix in ['.csv','.json']}}
    (out/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return report,ledger,state

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    run(parser.parse_args().output_dir)
