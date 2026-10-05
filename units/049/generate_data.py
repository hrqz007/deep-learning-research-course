"""Original bilingual teaching corpus; documents kept separate."""
from pathlib import Path
import argparse,json,hashlib
ROOT=Path(__file__).resolve().parent

def documents(start,n):
    materials=['iron','carbon','silicon','copper'];cn=['铁','碳','硅','铜'];texts=[]
    for i in range(n):
        k=start+i;j=i%4
        texts.append([f'sample {materials[j]} number {k} is stable.',f'样品{cn[j]}编号{k}保持稳定。',f'measure sample {materials[j]} at {280+k} K.',f'在{280+k} K测量{cn[j]}样品。'][i%4])
    return texts

def generate():
    train=documents(0,48);validation=documents(100,16);test=documents(200,22)+['新样品α含铁。','test sample 🧪 is new.']
    return {'train':train,'validation':validation,'test':test,'probes':['材料A','café','cafe\u0301','👩\u200d🔬','<PAD>','', 'new α 🧪'],
            'protocol':{'models':['byte','byte_bpe'],'bpe_merges':16,'embedding_dim':8,'steps':120,'learning_rate':.4,'seeds':[4911,4912,4913],'objective':'sum valid next-token NLL divided by count of nonpadding targets; EOS included','selection':'none; fixed tokenizer merge budget and optimizer; validation descriptive only','normalization':'identity; exact UTF-8 bytes preserved','boundaries':'BOS/EOS per document; no cross-document training pairs or BPE merges','pretraining':'none'}}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'data');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);raw=(json.dumps(generate(),ensure_ascii=False,indent=2)+'\n').encode();(a.output/'corpus.json').write_bytes(raw);(a.output/'data_manifest.json').write_text(json.dumps({'corpus_sha256':hashlib.sha256(raw).hexdigest(),'origin':'original synthetic bilingual scientific phrases; no personal data or external text'},indent=2)+'\n');print(len(raw),hashlib.sha256(raw).hexdigest())
