"""Unit 021: finite, offline evaluation. Numerical experiment uses only stdlib.

Scores are supplied sensor/10 values, not fitted or certified probabilities.
The run freezes its validation-only choice before first opening sealed_test.csv.
This pedagogical audit is not access control or a real preregistration system.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRID = (0.3, 0.4, 0.5, 0.7, 0.9)
EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)
OUTPUT_NAMES = ('frozen_plan.json', 'summary.json', 'thresholds.csv', 'calibration.csv')


DEFAULT_FIGURE_HASHES = {'data/sealed_test.csv': 'a06595f20f837685c329a331bdad5a7bdfbaf0b89d5c61b864b0bb99795f634d', 'data/split_panel.csv': '23c1512b0f8a20ca0eaefcc86609e9b56b28dbfea9c014e18cd54afedc0dc35a', 'data/train.csv': '40d92057ae824d6abb07682ae7d6d82433751184ae7680912d876e7d093a04f0', 'data/validation.csv': 'a7ae01ed086ad4fc9e50030c7603732d9f80a555ed3526c82f6d2d47913d46f8', 'outputs/calibration.csv': 'e2bf5da0b95505bed637127f00c165d0b88f7f9165245f5a2b57170efe4d08c1', 'outputs/frozen_plan.json': 'fd9f96afa4905a8a63f6bab2ccfd4fa81abb1a0d932d0776b8447310e26ea5d3', 'outputs/summary.json': 'd6794db1026864e1aee30133abb7781787f0a035919229c3d4371ad217e17c18', 'outputs/thresholds.csv': 'cfcb964ac90c3cda181492e74d57ab4403637e3657a7d0be19df9a452209c3e3'}

def require_default_figure_inputs():
    """Post-evaluation figure build only; deliberately not part of run's holdout flow."""
    for relative, expected in DEFAULT_FIGURE_HASHES.items():
        path = HERE / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Fixed teaching figures require original data and all four default outputs; custom runs must be saved separately.')


def paired(a, b):
    a, b = list(a), list(b)
    if not a or len(a) != len(b):
        raise ValueError('Inputs must have the same positive length.')
    return a, b


def real_number(x, name='value', low=None, high=None):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(name + ' must be a real int or float, not bool.')
    try:
        x = float(x)
    except OverflowError as exc:
        raise ValueError(name + ' is too large.') from exc
    if not math.isfinite(x) or (low is not None and x < low) or (high is not None and x > high):
        raise ValueError(name + ' is outside its finite range.')
    return x


def binary_labels(y):
    y = list(y)
    if not y or any(type(v) is not int or v not in (0, 1) for v in y):
        raise ValueError('Labels must be a nonempty sequence of integer 0 or 1.')
    return y


def binary_counts(y, prediction):
    y, prediction = paired(y, prediction)
    y, prediction = binary_labels(y), binary_labels(prediction)
    # Actual class is the row; predicted class is the column: [[TN, FP], [FN, TP]].
    c = [[0, 0], [0, 0]]
    for actual, predicted in zip(y, prediction):
        c[actual][predicted] += 1
    return {'tn': c[0][0], 'fp': c[0][1], 'fn': c[1][0], 'tp': c[1][1]}


def count_metrics(counts, zero_division=0):
    if type(zero_division) is not int or zero_division not in (0, 1):
        raise ValueError('zero_division must be integer 0 or 1.')
    if set(counts) != {'tn', 'fp', 'fn', 'tp'} or any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError('Counts must contain four nonnegative integers.')
    tn, fp, fn, tp = (counts[k] for k in ('tn', 'fp', 'fn', 'tp'))
    n = tn + fp + fn + tp
    if n == 0:
        raise ValueError('Empty evaluation set is not supported.')
    undefined = []
    def ratio(num, den, name):
        if den == 0:
            undefined.append(name)
            return float(zero_division)
        return num / den
    return {**counts, 'n': n, 'support': tp + fn, 'predicted_positive': tp + fp,
            'accuracy': (tp + tn) / n,
            'precision': ratio(tp, tp + fp, 'precision'),
            'recall': ratio(tp, tp + fn, 'recall'),
            'f1': ratio(2 * tp, 2 * tp + fp + fn, 'f1'),
            'undefined': undefined, 'zero_division': zero_division}


def classify(scores, threshold):
    threshold = real_number(threshold, 'threshold', 0, 1)
    scores = [real_number(v, 'score', 0, 1) for v in scores]
    if not scores:
        raise ValueError('Scores cannot be empty.')
    return [int(s >= threshold) for s in scores]


def binary_metrics(y, prediction, zero_division=0):
    return count_metrics(binary_counts(y, prediction), zero_division)


def threshold_table(y, scores, grid=GRID, c_fn=3, c_fp=1):
    y, scores = paired(y, scores)
    binary_labels(y)
    c_fn = real_number(c_fn, 'c_fn', 0)
    c_fp = real_number(c_fp, 'c_fp', 0)
    if c_fn + c_fp == 0:
        raise ValueError('At least one error cost must be positive.')
    grid = list(grid)
    if not grid:
        raise ValueError('Threshold grid cannot be empty.')
    grid = [real_number(t, 'threshold', 0, 1) for t in grid]
    if len(set(grid)) != len(grid):
        raise ValueError('Threshold grid cannot contain duplicates.')
    result = []
    for threshold in sorted(grid):
        row = binary_metrics(y, classify(scores, threshold))
        total = c_fn * row['fn'] + c_fp * row['fp']
        if not math.isfinite(total):
            raise ValueError('Cost overflow.')
        result.append({'threshold': threshold, **row, 'total_cost': total, 'mean_cost': total / len(y)})
    return result


def select_threshold(y, scores, grid=GRID):
    table = threshold_table(y, scores, grid)
    # Explicit, deterministic tie rule: lower total cost, then higher threshold.
    best = min(table, key=lambda r: (r['total_cost'], -r['threshold']))
    return best['threshold'], table


def multiclass_metrics(y, prediction, labels, zero_division=0):
    y, prediction = paired(y, prediction)
    labels = list(labels)
    if not labels or any(type(v) is not int for v in labels) or len(labels) != len(set(labels)):
        raise ValueError('labels must contain unique integers.')
    if any(type(v) is not int or v not in labels for v in y + prediction):
        raise ValueError('Every actual and predicted label must be in labels.')
    per_class = []
    for label in labels:
        m = binary_metrics([int(v == label) for v in y], [int(v == label) for v in prediction], zero_division)
        per_class.append({'label': label, **m})
    totals = {k: sum(r[k] for r in per_class) for k in ('tn', 'fp', 'fn', 'tp')}
    micro = count_metrics(totals, zero_division)
    # Only P/R/F1 of pooled one-vs-rest counts are meaningful micro outputs here.
    return {'labels': labels, 'n': len(y), 'per_class': per_class,
            'accuracy': sum(a == b for a, b in zip(y, prediction)) / len(y),
            'macro_f1': math.fsum(r['f1'] for r in per_class) / len(labels),
            'weighted_f1': math.fsum(r['support'] * r['f1'] for r in per_class) / len(y),
            'micro_precision': micro['precision'], 'micro_recall': micro['recall'], 'micro_f1': micro['f1'],
            'zero_division': zero_division}


def regression_metrics(y, prediction):
    y, prediction = paired(y, prediction)
    # Bounded teaching domain keeps squares and sums away from overflow.
    y = [real_number(v, 'target', -1e100, 1e100) for v in y]
    prediction = [real_number(v, 'prediction', -1e100, 1e100) for v in prediction]
    n = len(y)
    errors = [p - a for a, p in zip(y, prediction)]
    constant_target = all(v == y[0] for v in y)
    mean_y = y[0] if constant_target else math.fsum(v / n for v in y)
    mae = math.fsum(abs(e) / n for e in errors)
    def scaled_square(value):
        squared = value * value
        term = squared / n
        if value != 0 and (squared == 0 or term == 0):
            raise ValueError('Squared difference or averaged term underflows; rescale the data.')
        return term
    mse = math.fsum(scaled_square(e) for e in errors)
    variance = 0.0 if constant_target else math.fsum(scaled_square(v - mean_y) for v in y)
    r2 = None if n < 2 or variance == 0 else 1 - mse / variance
    if r2 is not None and not math.isfinite(r2):
        raise ValueError('R2 ratio is outside floating-point range.')
    return {'n': n, 'bias': math.fsum(e / n for e in errors), 'mae': mae, 'mse': mse,
            'rmse': math.sqrt(mse), 'r2': r2,
            'r2_undefined_reason': 'n<2 or constant target' if r2 is None else None}


def calibration_bins(y, probabilities, edges=EDGES):
    y, probabilities = paired(y, probabilities)
    y = binary_labels(y)
    probabilities = [real_number(v, 'probability', 0, 1) for v in probabilities]
    edges = [real_number(v, 'edge', 0, 1) for v in edges]
    if len(edges) < 2 or edges[0] != 0 or edges[-1] != 1 or any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError('Bin edges must strictly increase from 0 to 1.')
    rows = []
    for j, (left, right) in enumerate(zip(edges, edges[1:])):
        indexes = [i for i, p in enumerate(probabilities) if left <= p and (p < right or (j == len(edges) - 2 and p == 1))]
        count = len(indexes)
        mean_p = math.fsum(probabilities[i] for i in indexes) / count if count else None
        freq = sum(y[i] for i in indexes) / count if count else None
        rows.append({'left': left, 'right': right, 'right_closed': j == len(edges) - 2,
                     'n': count, 'positives': sum(y[i] for i in indexes), 'mean_probability': mean_p,
                     'positive_frequency': freq, 'absolute_gap': abs(mean_p - freq) if count else None})
    return {'bins': rows, 'ece': math.fsum(r['n'] * r['absolute_gap'] for r in rows if r['n']) / len(y),
            'brier': math.fsum((p - a) ** 2 for a, p in zip(y, probabilities)) / len(y),
            'note': 'Empirical positive-class bin gaps; not proof of population calibration.'}


def grouped_metrics(y, scores, groups, threshold):
    y, scores = paired(y, scores)
    groups = list(groups)
    if len(groups) != len(y) or any(not isinstance(g, str) or not g for g in groups):
        raise ValueError('Provide one nonempty group string per example.')
    predictions = classify(scores, threshold)
    binary_labels(y)
    result = {}
    for group in sorted(set(groups)):
        indexes = [i for i, g in enumerate(groups) if g == group]
        result[group] = binary_metrics([y[i] for i in indexes], [predictions[i] for i in indexes])
    return result


def load_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['row_id', 'entity', 'time', 'group', 'sensor', 'y']:
            raise ValueError('Unexpected data columns or column order.')
        rows = list(reader)
    if not rows:
        raise ValueError('Empty CSV.')
    out = []
    for row in rows:
        if None in row or any(v is None or v == '' for v in row.values()):
            raise ValueError('Missing or extra CSV fields.')
        item = {**row, 'time': int(row['time']), 'sensor': int(row['sensor']), 'y': int(row['y'])}
        if item['time'] < 1 or not 0 <= item['sensor'] <= 10 or item['y'] not in (0, 1):
            raise ValueError('CSV numeric value is outside its range.')
        out.append(item)
    if len({r['row_id'] for r in out}) != len(out):
        raise ValueError('Duplicate row IDs.')
    return out


def check_partitions(parts, require_entity=True, require_time=True):
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError('Three nonempty partitions are required.')
    ids = [set(r['row_id'] for r in part) for part in parts]
    entities = [set(r['entity'] for r in part) for part in parts]
    for i in range(3):
        if len(ids[i]) != len(parts[i]):
            raise ValueError('Duplicate row IDs in a partition.')
        for j in range(i + 1, 3):
            if ids[i] & ids[j]:
                raise ValueError('Row IDs overlap across partitions.')
            if require_entity and entities[i] & entities[j]:
                raise ValueError('Entities overlap across partitions.')
    if require_time and any(max(r['time'] for r in parts[i]) >= min(r['time'] for r in parts[i + 1]) for i in (0, 1)):
        raise ValueError('Partition times are not strictly ordered.')
    return {'row_counts': [len(p) for p in parts], 'entity_counts': [len(s) for s in entities],
            'entity_disjoint': all(not entities[i] & entities[j] for i in range(3) for j in range(i + 1, 3)),
            'strict_time_order': all(max(r['time'] for r in parts[i]) < min(r['time'] for r in parts[i + 1]) for i in (0, 1))}


def split_by_entity(rows, train_entities, validation_entities, test_entities):
    sets = [set(s) for s in (train_entities, validation_entities, test_entities)]
    if any(not s for s in sets) or any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Entity lists must be nonempty and disjoint.')
    if set.union(*sets) != {r['entity'] for r in rows}:
        raise ValueError('Entity lists must exactly cover observed entities.')
    parts = [[r for r in rows if r['entity'] in s] for s in sets]
    check_partitions(parts, require_entity=True, require_time=False)
    return parts


def split_by_time(rows, train_end, validation_end):
    if type(train_end) is not int or type(validation_end) is not int or train_end >= validation_end:
        raise ValueError('Time cutoffs must be increasing integers.')
    parts = [[r for r in rows if r['time'] <= train_end],
             [r for r in rows if train_end < r['time'] <= validation_end],
             [r for r in rows if r['time'] > validation_end]]
    check_partitions(parts, require_entity=False, require_time=True)
    return parts


def safe_output_dir(output_dir, input_dir):
    out, data = Path(output_dir).resolve(), Path(input_dir).resolve()
    if out == HERE or out in HERE.parents or out == data or data in out.parents or out in data.parents:
        raise ValueError('Output directory would overlap protected sources or input data.')
    if out.exists() and not out.is_dir():
        raise ValueError('Output destination is not a directory.')
    # Never follow a pre-existing output symlink, even if its target looks harmless.
    for name in OUTPUT_NAMES:
        dest = out / name
        if dest.is_symlink() or (dest.exists() and not dest.is_file()):
            raise ValueError('Output files must be regular files, not symlinks or directories.')
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def score_rows(rows):
    return [r['sensor'] / 10 for r in rows]


def make_plan(train, validation, input_dir):
    if {r['entity'] for r in train} & {r['entity'] for r in validation}:
        raise ValueError('Development entity overlap.')
    if {r['row_id'] for r in train} & {r['row_id'] for r in validation}:
        raise ValueError('Development row overlap.')
    if max(r['time'] for r in train) >= min(r['time'] for r in validation):
        raise ValueError('Development time overlap.')
    threshold, table = select_threshold([r['y'] for r in validation], score_rows(validation))
    plan = {'unit': '021', 'model': 'score = sensor / 10; fixed before data inspection; no fitting',
            'threshold': threshold, 'threshold_grid': list(GRID), 'decision': 'score >= threshold',
            'cost_false_negative': 3, 'cost_false_positive': 1, 'selection': 'validation mean cost, then larger threshold',
            'positive_label': 1, 'labels': [0, 1], 'zero_division': 0, 'calibration_edges': list(EDGES),
            'group_field': 'group', 'baseline_prediction': int(sum(r['y'] for r in train) / len(train) >= 0.5),
            'report': ['counts', 'accuracy', 'precision', 'recall', 'f1', 'mean_cost', 'by_group'],
            'development_sha256': {name: hashlib.sha256((Path(input_dir) / name).read_bytes()).hexdigest() for name in ('train.csv', 'validation.csv')},
            'scope': 'synthetic later records from held-out entities; descriptive results only'}
    return plan, table


def _run_staged(out, data):
    train, validation = load_rows(data / 'train.csv'), load_rows(data / 'validation.csv')
    plan, table = make_plan(train, validation, data)
    write_json(out / 'frozen_plan.json', plan)
    audit = ['read_train', 'read_validation', 'select_threshold_using_validation', 'write_frozen_plan']
    # First and only load of the sealed holdout in this run occurs below the freeze.
    test = load_rows(data / 'sealed_test.csv')
    audit.append('read_test_after_freeze')
    split = check_partitions([train, validation, test])
    test_y, test_scores = [r['y'] for r in test], score_rows(test)
    test_metrics = binary_metrics(test_y, classify(test_scores, plan['threshold']))
    test_metrics['mean_cost'] = (3 * test_metrics['fn'] + test_metrics['fp']) / len(test)
    matrix = [[8, 1, 1], [0, 1, 1], [0, 0, 1]]
    mc_y, mc_p = [], []
    for a, row in enumerate(matrix):
        for p, count in enumerate(row):
            mc_y.extend([a] * count); mc_p.extend([p] * count)
    cal = calibration_bins([0, 0, 1, 0, 1, 0, 1, 1], [0, .2, .25, .45, .5, .65, .75, 1])
    panel = load_rows(data / 'split_panel.csv')
    entity = split_by_entity(panel, ['A'], ['B'], ['C'])
    temporal = split_by_time(panel, 1, 2)
    summary = {'unit': '021', 'threshold': plan['threshold'], 'validation': next(r for r in table if r['threshold'] == plan['threshold']),
               'test': test_metrics, 'test_groups': grouped_metrics(test_y, test_scores, [r['group'] for r in test], plan['threshold']),
               'baseline_test': binary_metrics(test_y, [plan['baseline_prediction']] * len(test)),
               'partitions': split, 'audit': audit,
               'multiclass': multiclass_metrics(mc_y, mc_p, [0, 1, 2]),
               'multiclass_with_absent_label': multiclass_metrics(mc_y, mc_p, [0, 1, 2, 3]),
               'regression': regression_metrics([2, 4, 6, 8], [3, 4, 4, 9]),
               'calibration': cal,
               'panel_entity_split': check_partitions(entity, True, False),
               'panel_time_split': check_partitions(temporal, False, True),
               'sampling_caution': 'Repeated rows per entity are dependent; counts alone do not give confidence intervals.'}
    write_json(out / 'summary.json', summary)
    fields = ['threshold', 'tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'f1', 'accuracy', 'mean_cost']
    with (out / 'thresholds.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in table)
    with (out / 'calibration.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(cal['bins'][0])); writer.writeheader(); writer.writerows(cal['bins'])
    return summary


def run(output_dir=None, input_dir=None):
    data = Path(input_dir).resolve() if input_dir else HERE / 'data'
    out = safe_output_dir(output_dir if output_dir else HERE / 'outputs', data)
    # The current attempt's plan is physically written BEFORE reading its holdout.
    # A failed input/calculation never combines a new plan with old final results.
    with tempfile.TemporaryDirectory(prefix='.dl021-attempt-', dir=out.parent) as name:
        attempt = Path(name)
        summary = _run_staged(attempt, data)
        for filename in OUTPUT_NAMES:
            (attempt / filename).replace(out / filename)
    # This is not an atomic multi-file transaction against an OS I/O failure.
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--input-dir', type=Path)
    args = parser.parse_args()
    result = run(args.output_dir, args.input_dir)
    print(json.dumps({'threshold': result['threshold'], 'test_counts': {k: result['test'][k] for k in ('tn', 'fp', 'fn', 'tp')},
                      'test_mean_cost': result['test']['mean_cost'], 'test_f1': result['test']['f1']}, ensure_ascii=False, sort_keys=True))
