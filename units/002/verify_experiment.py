"""Author verification utility; advanced standard-library syntax is optional.

Checks are authored with the course, not an independent external review.
Run from any working directory: python /path/to/002/verify_experiment.py
"""
from pathlib import Path
from fractions import Fraction
import contextlib
import csv
import io
import json
import math
import runpy
import subprocess
import sys

BASE = Path(__file__).resolve().parent
checks = []

def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)

def read(column, filename):
    with (BASE / 'data' / filename).open(encoding='utf-8', newline='') as f:
        return [float(r[column]) for r in csv.DictReader(f)]

captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    result = runpy.run_path(str(BASE / 'experiment.py'))

expected = {
    'train_x': [1, 2, 3, 4, 5], 'train_y': [3, 5, 8, 9, 11],
    'candidates': [0, 1, 2], 'candidate_scores': [1.2, .2, .8],
    'best_bias': 1, 'best_score': .2,
    'a_train': [3, 5, 7, 9, 11], 'b_train': [3, 5, 8, 9, 11],
    'test_x': [1.5, 2.5, 3.5, 4.5], 'test_y': [4, 6, 8, 10],
    'stress_y': [5.5, 8.5, 11.5, 14.5],
    'a_test': [4, 6, 8, 10], 'b_test': [0, 0, 0, 0],
    'a_train_mae': .2, 'b_train_mae': 0, 'a_test_mae': 0,
    'b_test_mae': 7, 'a_stress_mae': 3,
}
for key, value in expected.items():
    check('expected_' + key, result[key] == value)
for var, filename, column in [('train_x','train.csv','x'),('train_y','train.csv','y'),
    ('test_x','test_inputs.csv','x'),('test_y','test_labels.csv','y'),
    ('stress_y','stress_labels.csv','y')]:
    check('csv_' + var, result[var] == read(column, filename))

# Exact rational reference from an explicitly hand-derived error table.
error_rows = [[1,1,2,1,1], [0,0,1,0,0], [1,1,0,1,1]]
for j, errors in enumerate(error_rows):
    exact = Fraction(sum(errors), 5)
    check('rational_candidate_' + str(j), result['candidate_scores'][j] == float(exact))
check('rational_test_B', result['b_test_mae'] == float(Fraction(28,4)))
check('rational_stress_A', result['a_stress_mae'] == float(Fraction(12,4)))
check('signed_error_counterexample', result['mae']([8,8],[7,9]) == 1)
check('lookup_seen_third', result['predict_b']([1,2,3],[3,5,8],3) == 8)
check('lookup_unseen', result['predict_b']([1,2,3],[3,5,8],2.5) == 0)
for name,actual,predicted in [('empty',[],[]),('mismatch',[1],[1,2])]:
    try:
        result['mae'](actual,predicted)
    except ValueError:
        check('reject_'+name, True)
    else:
        check('reject_'+name, False)

# Run changed-label cases as complete fresh scripts, not only helper functions.
source = (BASE/'experiment.py').read_text(encoding='utf-8')
for name, replacement, wanted_scores, wanted_bias in [
    ('remove_perturbation','[3, 5, 7, 9, 11]',[1.,0.,1.],1),
    ('tie','[2.5, 4.5, 6.5, 8.5, 10.5]',[.5,.5,1.5],0)]:
    changed = source.replace('train_y = [3, 5, 8, 9, 11]', 'train_y = '+replacement)
    changed += '\nprint("CHECK", candidate_scores, best_bias)\n'
    proc = subprocess.run([sys.executable,'-c',changed],capture_output=True,text=True,check=True)
    check(name, 'CHECK '+str(wanted_scores)+' '+str(wanted_bias) in proc.stdout)

for filename, error in [('01_name_error.py','NameError'),('02_type_error.py','TypeError'),
    ('03_index_error.py','IndexError'),('04_indent_error.py','IndentationError')]:
    proc = subprocess.run([sys.executable,str(BASE/'error_examples'/filename)],capture_output=True,text=True)
    check(filename, proc.returncode != 0 and error in proc.stderr)
for filename in ['05_reset_total.py','06_early_return.py']:
    proc = subprocess.run([sys.executable,str(BASE/'error_examples'/filename)],capture_output=True,text=True)
    check(filename, proc.returncode == 0 and proc.stdout.strip() == '0.0')

# The reference file intentionally omits the platform-dependent version line.
normal = subprocess.run([sys.executable,str(BASE/'experiment.py')],capture_output=True,text=True,check=True)
body = normal.stdout[normal.stdout.index('candidate 0'):]
check('expected_stdout', body == (BASE/'outputs/expected-result.txt').read_text())
repeat = subprocess.run([sys.executable,str(BASE/'experiment.py')],capture_output=True,text=True,check=True)
check('fresh_process_repeated_stdout', normal.stdout == repeat.stdout)
print(json.dumps({'unit':'002','status':'passed','python':sys.version.split()[0],
                  'check_count':len(checks),'checks':checks,
                  'scope':'Author checks; finite aligned synthetic measurements only; not external independent review'},
                 ensure_ascii=False,indent=2))
