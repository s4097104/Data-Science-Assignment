import math,pandas as pd
def dcg(rs): return sum(r/math.log2(i+2) for i,r in enumerate(rs))
def ndcg(rs,k=5):
    rs=list(rs[:k]); ideal=dcg(sorted(rs,reverse=True)); return dcg(rs)/ideal if ideal else 0.0
def evaluate(df,k=5):
    vals=[]
    for _,g in df.groupby('question_id'):
        g=g.sort_values('rank').head(k); vals.append(ndcg([int(a==b) for a,b in zip(g.retrieved_page,g.gold_page)],k))
    return sum(vals)/len(vals) if vals else 0
if __name__=='__main__':
    df=pd.read_csv('outputs/bm25_results.csv'); print('Starter page-match NDCG@5:',evaluate(df)); print('NOTE: page equality is only a proxy relevance judgment for this starter corpus.')
