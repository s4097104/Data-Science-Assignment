from collections import defaultdict
def reciprocal_rank_fusion(lists,weights=None,rrf_k=60,top_k=5):
    if weights is None: weights=[1.0]*len(lists)
    scores=defaultdict(float); payload={}
    for results,w in zip(lists,weights):
        for x in results:
            cid=x['chunk_id']; scores[cid]+=w/(rrf_k+x['rank']); payload[cid]=dict(x)
    ranked=sorted(scores.items(),key=lambda z:z[1],reverse=True)[:top_k]; out=[]
    for rank,(cid,s) in enumerate(ranked,1):
        x=payload[cid]; x.update(score=float(s),rank=rank,retriever='hybrid_rrf'); out.append(x)
    return out
