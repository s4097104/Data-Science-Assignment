import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from config import CHUNKS_FILE
from load_chunks import load_chunks
from bm25_retriever import bm25_search
from dense_retriever import dense_search
from hybrid_retriever import reciprocal_rank_fusion
chunks=load_chunks(CHUNKS_FILE); q=input('Enter a question: '); b=bm25_search(chunks,q,5); d=dense_search(chunks,q,5); h=reciprocal_rank_fusion([b,d],[.5,.5],top_k=5)
for title,rs in [('BM25',b),('DENSE',d),('HYBRID',h)]:
 print('\n'+title); [print(x['rank'],x['score'],x['page'],x['chunk_id']) for x in rs]
