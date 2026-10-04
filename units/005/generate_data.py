"""Create original, deterministic teaching fixtures. No downloads or real records."""
from pathlib import Path
import csv, json
ROOT = Path(__file__).resolve().parent

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def main():
    out = ROOT / 'data'; out.mkdir(exist_ok=True)
    records, labels = [], []
    offsets = {'A': -2, 'B': 2, 'C': -2, 'D': 2, 'E': 8, 'F': 10}
    for k, entity in enumerate(offsets):
        for day in (1, 2):
            x1 = 2*k + day; x2 = 10*x1
            y = 2*x1 + x2/10 + 1 + offsets[entity]
            rid = f'{entity}{day:02}'
            records.append({'record_id':rid, 'device_id':entity, 'predict_time':f'2026-01-{day:02}T09:00:00', 'knob1':x1, 'knob2':'' if rid in ['A02','D02'] else x2, 'post_reading':y, 'post_available_time':f'2026-01-{day:02}T09:05:00'})
            labels.append({'record_id':rid,'target':y,'label_available_time':f'2026-01-{day:02}T09:05:00'})
    write_csv(out/'records_raw.csv', records + [records[1].copy()])
    write_csv(out/'labels.csv', list(reversed(labels)))
    entity_split={'train':['A01','A02','B01','B02','C01','C02'],'validation':['D01','D02'],'test':['E01','E02','F01','F02']}
    time_split={'train':[f'{e}01' for e in 'ABCDEF'],'validation':[f'{e}02' for e in 'ABCDEF']}
    (out/'entity_split.json').write_text(json.dumps(entity_split,indent=2)+'\n')
    (out/'time_illustration.json').write_text(json.dumps({'purpose':'illustration only: day 3 test records do not exist in this fixture','train':time_split['train'],'validation':time_split['validation'],'test_prediction_date':'2026-01-03','test_records_available':False},indent=2)+'\n')
    memory=[{'record_id':f'{e}{t}', 'device_id':e,'visit':t,'target':v} for e,v in zip('ABCDEF',[2,5,11,17,29,43]) for t in (1,2)]
    write_csv(out/'memorization.csv',memory)
    print('Created 13 raw rows, 12 unique observations, 12 reversed labels, fixed entity lists.')
if __name__ == '__main__':
    main()
