"""Independent inference replay from saved model/data arrays; no retraining."""
from pathlib import Path
import argparse,json
import numpy as np
import experiment as e

def main(output='outputs'):
    out=Path(output);data=np.load(out/'data.npz');weights=np.load(out/'weights.npz');preds=np.load(out/'predictions.npz');metrics=json.loads((out/'metrics.json').read_text())
    bands=e.band_features(e.stft(data['waves']));np.testing.assert_allclose(bands,data['bands'],atol=1e-12)
    test=data['split']==2;train=data['split']==0;maximum=0.;checks=0
    for row in metrics['runs']:
        method,seed=row['method'],row['seed'];key=f'{method}_{seed}';model={name:weights[key+'_'+name] for name in ['w','mu','sd']}
        x=e.features(bands,method!='pooled');xs=e.features(e.shuffle_frames(bands,seed+9000),method!='pooled')
        fitted=xs if method=='shuffle_train' else x
        np.testing.assert_allclose(model['mu'],fitted[train].mean(0),atol=1e-12)
        p=e.predict(model,x[test]);ps=e.predict(model,xs[test])
        for pp,name,field in [(p,key,'test_accuracy'),(ps,key+'_shuffled','shuffle_test_accuracy')]:
            np.testing.assert_allclose(pp,preds[name],atol=1e-12);maximum=max(maximum,float(np.max(np.abs(pp-preds[name]))))
            np.testing.assert_allclose(np.mean((pp>=.5)==data['labels'][test]),row[field],atol=1e-12);checks+=1
    report={'status':'passed','probability_arrays_replayed':checks,'max_probability_error':maximum,'feature_reconstruction':'passed','training_only_mean':'passed'}
    (out/'replay-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs');main(p.parse_args().output)
