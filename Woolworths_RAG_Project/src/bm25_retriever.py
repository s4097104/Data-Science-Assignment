import math,re
from collections import Counter
TOKEN_RE=re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
def tokenize(t): return TOKEN_RE.findall(t.lower())
class BM25:
    def __init__(self,docs,k1=1.5,b=0.75):
        self.docs=docs; self.k1=k1; self.b=b; self.tokens=[tokenize(x) for x in docs]; self.lens=[len(x) for x in self.tokens]; self.avg=sum(self.lens)/max(len(self.lens),1); self.df=Counter()
        for ts in self.tokens:
            for term in set(ts): self.df[term]+=1
    def score(self,q,i):
        qterms=tokenize(q); tf=Counter(self.tokens[i]); dl=self.lens[i]; N=len(self.docs); s=0.0
        for term in qterms:
            if term not in self.df: continue
            df=self.df[term]; idf=math.log(1+(N-df+0.5)/(df+0.5)); f=tf[term]
            den=f+self.k1*(1-self.b+self.b*dl/max(self.avg,1e-9))
            s+=idf*((f*(self.k1+1))/max(den,1e-9))
        return s
    def search(self,q,k=5): return sorted([(i,self.score(q,i)) for i in range(len(self.docs))],key=lambda x:x[1],reverse=True)[:k]
def bm25_search(chunks,q,k=5):
    r=BM25([c['text'] for c in chunks]); out=[]
    for rank,(i,s) in enumerate(r.search(q,k),1):
        x=dict(chunks[i]); x.update(score=float(s),rank=rank,retriever='bm25'); out.append(x)
    return out
