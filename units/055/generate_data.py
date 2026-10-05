"""256 unique grammar strings over a declared alphabet, split once before fitting."""
from pathlib import Path
import itertools,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
VOCAB=['<pad>','<bos>','<eos>','蓝','红','白','绿','猫','狗','鸟','鱼','看','追','找','拿','球','云','书','花','。']
PAD,BOS,EOS=0,1,2

def corpus():
    rows=[]
    for c,a,v,o in itertools.product(range(4),repeat=4):
        # Color2 depends on THREE earlier categorical choices; object is a free choice.
        tokens=[3+c,7+a,11+v,3+(c+a+v)%4,15+o,19]
        rows.append({'id':f'{c}{a}{v}{o}','tokens':tokens,'text':' '.join(VOCAB[t] for t in tokens)})
    order=np.random.default_rng(5500).permutation(len(rows))
    for i,index in enumerate(order):
        rows[index]['split']='train' if i<128 else ('validation' if i<192 else 'test')
        rows[index]['split_order']=i
    return sorted(rows,key=lambda r:r['split_order'])
def main(output=ROOT/'data'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    obj={'vocab':VOCAB,'rule':'color2=(color1_index+animal_index+verb_index) mod4; 4 free choices, each4; punctuation fixed',
         'split_seed':5500,'documents':corpus(),'vocab_policy':'predeclared synthetic alphabet'}
    p=out/'corpus.json';p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps({'corpus.json':{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}},indent=2)+'\n');return obj
if __name__=='__main__':main()
