"""Reproducible diagnostic evaluation. No LLM or dense model is invoked.
Run: python evaluate.py   (requires numpy and scikit-learn)
"""
import csv,json,sys,math,hashlib,platform
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'audit_inputs/src'))
from bm25_retriever import BM25
cs=list(csv.DictReader(open(P/'audit_inputs/data/chunks.csv',encoding='utf-8-sig')))
qs=json.loads((P/'benchmark.json').read_text());O=P/'new_evidence';O.mkdir(exist_ok=True)
texts=[c['text'] for c in cs]
bm=BM25(texts); no_norm=BM25(texts,b=0)
v=TfidfVectorizer(lowercase=True,ngram_range=(1,2));X=v.fit_transform(texts)
def rank(scores):return sorted(range(len(cs)),key=lambda i:(-float(scores[i]),i))
def scores(q):
 a=np.array([bm.score(q,i) for i in range(len(cs))]);b=np.array([no_norm.score(q,i) for i in range(len(cs))]);t=cosine_similarity(v.transform([q]),X)[0]
 ar=rank(a)[:5];tr=rank(t)[:5];rrf=np.zeros(len(cs))
 for li in [ar,tr]:
  for j,i in enumerate(li,1):rrf[i]+=.5/(60+j)
 return {'BM25':a,'BM25_b0':b,'TFIDF':t,'Lexical_RRF':rrf}
def metric(order,gold,k):
 rel=[int(cs[i]['chunk_id'] in gold) for i in order[:k]]
 ideal=sum(1/math.log2(i+2) for i in range(min(k,len(gold))))
 return {'hit':int(any(rel)),'recall':sum(rel)/len(gold),'mrr':next((1/(i+1) for i,r in enumerate(rel) if r),0),'ndcg':sum(r/math.log2(i+2) for i,r in enumerate(rel))/ideal}
rows=[];raw=[];screen=[]
for q in qs:
 for name,sc in scores(q['question']).items():
  order=rank(sc)
  for j,i in enumerate(order[:5],1):raw.append(dict(question_id=q['id'],type=q['type'],method=name,rank=j,chunk_id=cs[i]['chunk_id'],page=cs[i]['page'],score=float(sc[i])))
  if q['answerable']:
   for k in [1,3,5]:rows.append(dict(question_id=q['id'],type=q['type'],method=name,k=k,**metric(order,set(q['relevant_chunks']),k)))
  if name in ['BM25','TFIDF']:
   screen.append(dict(question_id=q['id'],type=q['type'],method=name,zero_overlap=bool(max(sc)<=0)))
def write(f,rr):
 with open(O/f,'w',newline='') as o:
  w=csv.DictWriter(o,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
write('rankings.csv',raw);write('metrics.csv',rows);write('zero_overlap_screen.csv',screen)
summary=[]
for name in ['BM25','BM25_b0','TFIDF','Lexical_RRF']:
 for kind in ['all','direct','paraphrase','comparison']:
  for k in [1,3,5]:
   rr=[r for r in rows if r['method']==name and r['k']==k and (kind=='all' or r['type']==kind)]
   summary.append(dict(method=name,type=kind,k=k,n=len(rr),**{m:sum(r[m] for r in rr)/len(rr) for m in ['hit','recall','mrr','ndcg']}))
write('summary.csv',summary)
# Existing original 18 questions are the development baseline; run the ablation on those too.
old=list(csv.DictReader(open(P/'audit_inputs/data/evaluation_questions.csv',encoding='utf-8-sig')));dev=[]
for q in old:
 gold={c['chunk_id'] for c in cs if c['page']==q['expected_source_page']}
 for name,sc in scores(q['question']).items():dev.append(dict(id=q['question_id'],method=name,**metric(rank(sc),gold,5)))
write('original18_ablation.csv',dev)
# Frozen, author-reviewed answer arithmetic; these are reference checks, not model answers.
checks=[('EBIT',3223,2754,-469),('revenue',67922,69077,1155),('debt',15504,15989,485),('cash',4359,4550,191),('dividend',144,84,-60),('EPS',140.3,113.5,-26.8),('profit',108,963,855)]
calc=[]
for name,a,b,d in checks:
 assert abs((b-a)-d)<1e-8
 calc.append(dict(measure=name,prior=a,current=b,change=round(b-a,4),percent_change=round((b-a)/a*100,4)))
write('reference_arithmetic.csv',calc)
assert metric([0,2,3,4,5],{'chunk_001','chunk_002'},5)['ndcg']<1
assert len({q['id'] for q in qs})==40
assert all(set(q['relevant_chunks']) <= {c['chunk_id'] for c in cs} for q in qs)
(O/'checks.txt').write_text('PASS unique question IDs, valid gold chunk IDs, seven arithmetic checks, missing-relevant-passage NDCG check.\n')
manifest={'python':platform.python_version(),'numpy':np.__version__,'parameters':{'BM25_k1':1.5,'BM25_b':.75,'ablation_b':0,'TFIDF':'word unigrams+bigrams, default sklearn settings','Lexical_RRF':'equal weights, constant 60, top 5 per input'},'hashes':{str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'benchmark.json',P/'evaluate.py',P/'audit_inputs/data/chunks.csv']}}
(O/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps([x for x in summary if x['type']=='all'],indent=2))
print('Unsupported zero-score detections:',[(name,sum(r['zero_overlap'] for r in screen if r['method']==name and r['type']=='unsupported')) for name in ['BM25','TFIDF']])
