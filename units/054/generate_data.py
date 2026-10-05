"""Original closed-vocabulary toy corpus; no downloads or private data."""
from pathlib import Path
import json,hashlib,unicodedata
ROOT=Path(__file__).resolve().parent
VOCAB=['<pad>','<bos>','<eos>','<mask>','蓝','红','白','猫','狗','鸟','追','看','球','云','。']
RAW=['蓝 猫 追 红 球','红 狗 追 蓝 球','蓝 鸟 看 白 云','白 猫 看',
     '红 鸟 追 白 云 。','白 狗 看 球','蓝 猫 看 白 云','红 狗 看 红 球',
     '蓝   猫 追 红 球','白 鸟 追 蓝 球','蓝 狗 追 白 云']
def normalize(text):
    # Only NFC and whitespace normalization. No claim of semantic near-deduplication.
    return ' '.join(unicodedata.normalize('NFC',text).split())
def corpus():
    seen={};rows=[];duplicates=[]
    for i,text in enumerate(RAW):
        cleaned=normalize(text);key=hashlib.sha256(cleaned.encode()).hexdigest()
        if key in seen:
            duplicates.append({'removed':f'd{i:02}','kept':seen[key],'normalized_sha256':key});continue
        seen[key]=f'd{i:02}'
        index=len(rows);split='train' if index<6 else ('validation' if index<8 else 'test')
        rows.append({'id':f'd{i:02}','text':cleaned,'tokens':[VOCAB.index(t) for t in cleaned.split()],
                     'split':split,'normalized_sha256':key})
    return rows,duplicates

def main(output=ROOT/'data'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);rows,dups=corpus()
    obj={'vocabulary':VOCAB,'vocabulary_policy':'predeclared closed synthetic alphabet, not fit on held-out text',
         'raw':[{'id':f'd{i:02}','text':t} for i,t in enumerate(RAW)],'documents':rows,'duplicates':dups,
         'split_policy':'deduplicate exact NFC+whitespace normalized texts; first6 train, next2 validation, last2 test'}
    p=out/'corpus.json';p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps({'corpus.json':{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}},indent=2)+'\n')
    return obj
if __name__=='__main__':main()
