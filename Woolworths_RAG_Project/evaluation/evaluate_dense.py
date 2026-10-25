"""CPU ONNX evaluation of the prototype's embedding model.
python evaluate_dense.py /path/to/model_dir
Directory requires model.onnx, tokenizer.json, config.json from
sentence-transformers/multi-qa-MiniLM-L6-cos-v1 (onnx/model.onnx).
Mean pooling + L2 normalisation; truncation at 512 tokens. No LLM is run.
"""
import numpy as np,sys,json,csv,hashlib
from pathlib import Path
# Optional dependency directory for this audit environment only.
if len(sys.argv)>2:sys.path.append(sys.argv[2])
import onnxruntime as ort
from tokenizers import Tokenizer
P=Path(__file__).resolve().parent;M=Path(sys.argv[1]);sys.path.insert(0,str(P/'audit_inputs/src'))
from bm25_retriever import BM25
cs=list(csv.DictReader(open(P/'audit_inputs/data/chunks.csv',encoding='utf-8-sig')));qs=json.loads((P/'benchmark.json').read_text());O=P/'new_evidence';O.mkdir(exist_ok=True)
tok=Tokenizer.from_file(str(M/'tokenizer.json'))
tok.no_truncation()
tok.no_padding()
lengths=[len(tok.encode(c['text']).ids) for c in cs]
tok.enable_truncation(max_length=512);tok.enable_padding(pad_id=0,pad_token='[PAD]')
sess=ort.InferenceSession(str(M/'model.onnx'),providers=['CPUExecutionProvider'])
def encode(texts):
 batches=[]
 for start in range(0,len(texts),8):
  enc=tok.encode_batch(texts[start:start+8]);ids=np.array([x.ids for x in enc],dtype=np.int64);mask=np.array([x.attention_mask for x in enc],dtype=np.int64);types=np.array([x.type_ids for x in enc],dtype=np.int64)
  hs=sess.run(None,{'input_ids':ids,'attention_mask':mask,'token_type_ids':types})[0]
  v=(hs*mask[:,:,None]).sum(axis=1)/mask.sum(axis=1)[:,None];v=v/np.linalg.norm(v,axis=1,keepdims=True);batches.append(v)
 return np.concatenate(batches)
D=encode([x['text'] for x in cs]);Q=encode([x['question'] for x in qs]);bm=BM25([x['text'] for x in cs])
def rank(a):return sorted(range(len(cs)),key=lambda i:(-float(a[i]),i))
def measures(order,gold,k):
 rel=[int(cs[i]['chunk_id'] in gold) for i in order[:k]];ideal=sum(1/np.log2(i+2) for i in range(min(k,len(gold))))
 return dict(hit=int(any(rel)),recall=sum(rel)/len(gold),mrr=next((1/(j+1) for j,v in enumerate(rel) if v),0),ndcg=sum(v/np.log2(j+2) for j,v in enumerate(rel))/ideal)
raw=[];rows=[]
for qi,q in enumerate(qs):
 ds=D@Q[qi];br=rank([bm.score(q['question'],i) for i in range(len(cs))])[:5];dr=rank(ds)[:5];rr=np.zeros(len(cs))
 for li in [br,dr]:
  for j,i in enumerate(li,1):rr[i]+=.5/(60+j)
 for name,sc in [('Dense_ONNX',ds),('Hybrid_BM25_Dense',rr)]:
  order=rank(sc)
  for j,i in enumerate(order[:5],1):raw.append(dict(question_id=q['id'],type=q['type'],method=name,rank=j,chunk_id=cs[i]['chunk_id'],page=cs[i]['page'],score=float(sc[i])))
  if q['answerable']:
   for k in [1,3,5]:rows.append(dict(question_id=q['id'],type=q['type'],method=name,k=k,**measures(order,set(q['relevant_chunks']),k)))
def write(f,rs):
 with open(O/f,'w',newline='') as o:w=csv.DictWriter(o,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
write('dense_rankings.csv',raw);write('dense_metrics.csv',rows)
summary=[]
for name in ['Dense_ONNX','Hybrid_BM25_Dense']:
 for typ in ['all','direct','paraphrase','comparison']:
  for k in [1,3,5]:
   r=[x for x in rows if x['method']==name and x['k']==k and (typ=='all' or x['type']==typ)]
   summary.append(dict(method=name,type=typ,k=k,n=len(r),**{m:sum(x[m] for x in r)/len(r) for m in ['hit','recall','mrr','ndcg']}))
write('dense_summary.csv',summary);write('token_lengths.csv',[dict(chunk_id=c['chunk_id'],tokens_before_truncation=n,truncated=n>512) for c,n in zip(cs,lengths)])
(O/'dense_manifest.json').write_text(json.dumps({'model':'sentence-transformers/multi-qa-MiniLM-L6-cos-v1','runtime':'ONNX CPU','ort_version':ort.__version__,'pooling':'attention-mask mean then L2','max_tokens':512,'hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in M.iterdir() if f.is_file()}},indent=2))
print(json.dumps([s for s in summary if s['type']=='all'],indent=2));print('truncated',sum(n>512 for n in lengths),lengths)
