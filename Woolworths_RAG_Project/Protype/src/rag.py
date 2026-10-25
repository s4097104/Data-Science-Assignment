from config import CHUNKS_FILE,TOP_K,BM25_WEIGHT,DENSE_WEIGHT
from load_chunks import load_chunks
from bm25_retriever import bm25_search
from dense_retriever import dense_search
from hybrid_retriever import reciprocal_rank_fusion
from ollama_client import generate_answer
class RAGPipeline:
    def __init__(self,k=TOP_K): self.chunks=load_chunks(CHUNKS_FILE); self.k=k
    def retrieve(self,q):
        b=bm25_search(self.chunks,q,self.k); d=dense_search(self.chunks,q,self.k); h=reciprocal_rank_fusion([b,d],[BM25_WEIGHT,DENSE_WEIGHT],top_k=self.k); return b,d,h
    def answer(self,q):
        b,d,h=self.retrieve(q); return {'question':q,'answer':generate_answer(q,h),'bm25':b,'dense':d,'hybrid':h}
