"""DL090: reproducible CPU-only temporal audio experiment (NumPy)."""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/dl090-matplotlib')
import argparse, csv, json, platform, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FS, N, WINDOW, HOP = 8000, 4096, 256, 64
SEEDS = [11, 23, 37]


def generate(seed=9001, sources=40, pairs=6):
    """Each source supplies AB/BA pairs; a pair shares its two wave chunks."""
    rng = np.random.default_rng(seed)
    waves, labels, groups, pair_ids = [], [], [], []
    t = np.arange(N // 2) / FS
    envelope = np.sin(np.pi * np.arange(N // 2) / (N // 2 - 1)) ** 2
    for source in range(sources):
        offset, gain = rng.uniform(-80, 80), rng.uniform(.65, 1.35)
        for pair in range(pairs):
            chunks = []
            for f in [800 + offset, 1600 + offset]:
                phase = rng.uniform(0, 2 * np.pi)
                chunk = gain * envelope * np.sin(2 * np.pi * f * t + phase)
                chunk += rng.normal(0, .12, len(t))
                chunks.append(chunk)
            for y in [0, 1]:
                waves.append(np.concatenate(chunks if y == 0 else chunks[::-1]))
                labels.append(y); groups.append(source); pair_ids.append(source * pairs + pair)
    return np.array(waves), np.array(labels), np.array(groups), np.array(pair_ids)


def stft(waves):
    x = np.asarray(waves)
    if x.ndim != 2 or x.shape[1] < WINDOW:
        raise ValueError('Expected [batch, samples] with samples >= WINDOW')
    frames = np.lib.stride_tricks.sliding_window_view(x, WINDOW, axis=1)[:, ::HOP, :]
    return np.fft.rfft(frames * np.hanning(WINDOW), axis=-1)


def band_features(spec):
    f = np.fft.rfftfreq(WINDOW, 1 / FS)
    power = np.abs(spec) ** 2 / np.sum(np.hanning(WINDOW) ** 2)
    bands = np.stack([power[..., (f >= lo) & (f < hi)].sum(-1)
                      for lo, hi in [(550, 1100), (1300, 1900)]], axis=-1)
    # A per-frame quantity; no statistics from validation/test are fitted here.
    return np.log1p(bands)


def features(bands, temporal=True):
    if temporal:
        return np.concatenate([part.mean(1) for part in np.array_split(bands, 4, axis=1)], axis=1)
    return bands.mean(1)


def shuffle_frames(bands, seed):
    rng = np.random.default_rng(seed)
    return np.stack([row[rng.permutation(len(row))] for row in bands])


def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1 / (1 + np.exp(-z))


def fit_logistic(x, y, seed=11, steps=600):
    mu, sd = x.mean(0), x.std(0)
    sd = np.maximum(sd, 1e-8)
    z = np.column_stack(((x - mu) / sd, np.ones(len(x))))
    rng = np.random.default_rng(seed)
    w = rng.normal(0, .01, z.shape[1]); losses = []
    for step in range(steps):
        p = sigmoid(z @ w)
        reg = .001 * w; reg[-1] = 0
        grad = z.T @ (p - y) / len(y) + reg
        w -= .15 * grad
        if step % 10 == 0 or step == steps - 1:
            losses.append(float(np.mean(np.logaddexp(0, z @ w) - y * (z @ w)) + .0005 * np.sum(w[:-1] ** 2)))
    return {'w': w, 'mu': mu, 'sd': sd, 'losses': losses}


def predict(model, x):
    z = np.column_stack(((x - model['mu']) / model['sd'], np.ones(len(x))))
    return sigmoid(z @ model['w'])


def alignment_report():
    starts = np.arange(0, N - WINDOW + 1, HOP)
    centers = (starts + (WINDOW - 1) / 2) / FS
    event_start, event_end = N / 2 / FS, N / FS
    overlap = np.maximum(0, np.minimum((starts + WINDOW) / FS, event_end) - np.maximum(starts / FS, event_start))
    coverage = overlap / (WINDOW / FS)
    return starts, centers, coverage


def make_figures(out):
    out = Path(out); figs = out.parent / 'figures'; figs.mkdir(exist_ok=True)
    data = np.load(out / 'data.npz'); results = json.loads((out / 'metrics.json').read_text())
    x = data['waves']; t = np.arange(N) / FS
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for y in [0, 1]:
        idx = np.flatnonzero(data['labels'] == y)[0]
        ax[y, 0].plot(t, x[idx], lw=.5); ax[y, 0].set(xlabel='Time (s)', ylabel='Amplitude', title=['Low then high', 'High then low'][y])
        power = 10 * np.log10(np.abs(stft(x[idx:idx+1])[0].T) ** 2 + 1e-6)
        ax[y, 1].imshow(power, origin='lower', aspect='auto', extent=[(WINDOW-1)/(2*FS)-HOP/(2*FS), (N-WINDOW+(WINDOW-1)/2)/FS+HOP/(2*FS), -FS/(2*WINDOW), FS/2+FS/(2*WINDOW)], cmap='magma')
        ax[y, 1].set(xlabel='Frame center (s)', ylabel='Frequency (Hz)', ylim=(0, 2300), title='Hann STFT power (dB, relative)')
    fig.savefig(figs / '01_audio_spectrogram.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    tt = np.arange(32) / 8000
    ax[0].plot(tt * 1000, np.sin(2*np.pi*1000*tt), 'o-', label='1000 Hz sampled')
    ax[0].plot(tt * 1000, -np.sin(2*np.pi*7000*tt), 'x--', label='7000 Hz, reversed phase')
    ax[0].set(xlabel='Time (ms)', ylabel='Amplitude', title='Different signals, identical samples'); ax[0].legend(fontsize=8)
    u = np.arange(WINDOW); signal = np.sin(2*np.pi*830*u/FS)
    for window, name in [(np.ones(WINDOW), 'Rectangular'), (np.hanning(WINDOW), 'Hann')]:
        a = np.abs(np.fft.rfft(signal * window, n=8192)); a /= a.max()
        ax[1].plot(np.fft.rfftfreq(8192, 1/FS), 20*np.log10(np.maximum(a, 1e-6)), label=name)
    ax[1].set(xlim=(400, 1300), ylim=(-80, 2), xlabel='Frequency (Hz)', ylabel='Normalized magnitude (dB)', title='Main-lobe / side-lobe tradeoff'); ax[1].legend()
    fig.savefig(figs / '02_sampling_window.png', dpi=160); plt.close(fig)
    starts, centers, coverage = alignment_report()
    fig, ax = plt.subplots(2, 1, figsize=(10, 4.8), constrained_layout=True)
    for m in [27, 28, 29, 30, 31, 32]:
        ax[0].plot([starts[m]/FS, (starts[m]+WINDOW)/FS], [m, m], lw=6)
        ax[0].plot(centers[m], m, 'ko', ms=3)
    ax[0].axvline(.256, color='r', ls='--', label='Event boundary'); ax[0].set(xlabel='Time (s)', ylabel='Frame index', title='Frames overlap an event boundary'); ax[0].legend()
    ax[1].plot(centers, coverage, 'o-', ms=3, label='Fraction overlapping second event')
    ax[1].axvline(.256, color='r', ls='--'); ax[1].set(xlabel='Frame center (s)', ylabel='Overlap fraction', ylim=(-.05,1.05)); ax[1].legend()
    fig.savefig(figs / '03_label_alignment.png', dpi=160); plt.close(fig)
    rows = results['runs']; methods = ['temporal', 'pooled', 'shuffle_train']
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for j, field in enumerate(['test_accuracy', 'shuffle_test_accuracy']):
        values = [np.mean([r[field] for r in rows if r['method']==m]) for m in methods]
        ax[j].bar(methods, values, color=['#16697a','#e29578','#83c5be']); ax[j].axhline(.5, ls='--', c='gray')
        ax[j].set(ylim=(0,1.05), ylabel='Accuracy', title=['Ordered held-out sources', 'Frame-shuffle diagnostic'][j]); ax[j].tick_params(axis='x', labelrotation=15)
    fig.savefig(figs / '04_temporal_controls.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    src = data['groups']; split = data['split']; counts = np.zeros((40,3))
    for g in range(40):
        for s in range(3): counts[g,s] = np.sum((src==g)&(split==s))
    ax[0].imshow(counts.T, aspect='auto', cmap='Blues'); ax[0].set(yticks=[0,1,2], yticklabels=['Train','Validation','Test'], xlabel='Source ID', title='Group split: no source crosses roles')
    for row in rows:
        if row['seed']==11: ax[1].plot(np.arange(len(row['losses']))*10, row['losses'], label=row['method'])
    ax[1].set(xlabel='Update (last point approx.)', ylabel='Training objective', title='Fixed-budget logistic training'); ax[1].legend()
    fig.savefig(figs / '05_split_training.png', dpi=160); plt.close(fig)


def main(output='outputs'):
    t0=time.perf_counter(); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    waves,y,groups,pairs=generate(); spec=stft(waves); bands=band_features(spec)
    split=np.where(groups<24,0,np.where(groups<32,1,2)); tr,va,te=[split==s for s in range(3)]
    np.savez_compressed(out/'data.npz',waves=waves,labels=y,groups=groups,pairs=pairs,split=split,bands=bands)
    runs=[]; preds={}; weights={}
    for seed in SEEDS:
        shuffled=shuffle_frames(bands,seed+9000)
        for method in ['temporal','pooled','shuffle_train']:
            temporal=method!='pooled'
            ordered_x=features(bands,temporal); shuffled_x=features(shuffled,temporal)
            xtrain=shuffled_x if method=='shuffle_train' else ordered_x
            model=fit_logistic(xtrain[tr],y[tr],seed)
            p=predict(model,ordered_x[te]); ps=predict(model,shuffled_x[te]); pv=predict(model,ordered_x[va])
            key=f'{method}_{seed}'; preds[key]=p; preds[key+'_shuffled']=ps
            for name in ['w','mu','sd']: weights[key+'_'+name]=model[name]
            group_acc={str(g):float(np.mean((p[groups[te]==g]>=.5)==y[te][groups[te]==g])) for g in np.unique(groups[te])}
            runs.append({'method':method,'seed':seed,'test_accuracy':float(np.mean((p>=.5)==y[te])),
                         'shuffle_test_accuracy':float(np.mean((ps>=.5)==y[te])), 'validation_accuracy':float(np.mean((pv>=.5)==y[va])),
                         'group_accuracy':group_acc,'losses':model['losses']})
    np.savez_compressed(out/'predictions.npz',labels=y[te],groups=groups[te],**preds)
    np.savez_compressed(out/'weights.npz',**weights)
    rng=np.random.default_rng(9020); random_order=rng.permutation(len(y)); random_train=random_order[:288]; random_test=random_order[384:]
    common=sorted(set(groups[random_train])&set(groups[random_test]))
    starts,centers,overlap=alignment_report()
    with (out/'alignment.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['frame','start_sample','center_s','second_event_overlap']); w.writerows(zip(range(len(starts)),starts,centers,overlap))
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['method','seed','validation_accuracy','test_accuracy','shuffle_test_accuracy']);w.writeheader()
        for r in runs:w.writerow({k:r[k] for k in w.fieldnames})
    results={'unit':'090','data_seed':9001,'seeds':SEEDS,'fs':FS,'samples':N,'window':WINDOW,'hop':HOP,
             'stft_shape':list(spec.shape),'split_samples':[int(tr.sum()),int(va.sum()),int(te.sum())],
             'split_sources':[24,8,8],'group_overlap':0,'random_clip_split_source_overlap':len(common),
             'pooled_shuffle_max_error':float(np.max(np.abs(features(bands,False)-features(shuffle_frames(bands,99),False)))),
             'alias_max_error':float(np.max(np.abs(np.sin(2*np.pi*1000*np.arange(100)/FS)+np.sin(2*np.pi*7000*np.arange(100)/FS)))),
             'runs':runs,'environment':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},
             'elapsed_seconds_before_plotting':time.perf_counter()-t0,
             'limitations':['Synthetic two-tone task; not speech recognition','Only 8 held-out sources','Shuffling is a destructive intervention, not a valid natural distribution','Seed SD is not population confidence']}
    (out/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8');make_figures(out)
    print(json.dumps({k:results[k] for k in ['stft_shape','group_overlap','random_clip_split_source_overlap','elapsed_seconds_before_plotting']}))
    for r in runs:print(r['method'],r['seed'],r['test_accuracy'],r['shuffle_test_accuracy'])
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs');main(p.parse_args().output)
