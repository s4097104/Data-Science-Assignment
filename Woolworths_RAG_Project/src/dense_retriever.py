from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL,OUTPUT_DIR

def build_embeddings(chunks):
    model=SentenceTransformer(EMBEDDING_MODEL); texts=[c['text'] for c in chunks]
    if hasattr(model,'encode_document'): emb=model.encode_document(texts,normalize_embeddings=True,show_progress_bar=True)
    else: emb=model.encode(texts,normalize_embeddings=True,show_progress_bar=True)
    path=OUTPUT_DIR/'dense_embeddings.npy'; np.save(path,np.asarray(emb)); return path

def dense_search(chunks,q,k=5):
    path=OUTPUT_DIR/'dense_embeddings.npy'
    if not path.exists(): raise FileNotFoundError('Run python src/build_embeddings.py first.')
    model=SentenceTransformer(EMBEDDING_MODEL); emb=np.load(path)
    if hasattr(model,'encode_query'): qe=model.encode_query([q],normalize_embeddings=True)[0]
    else: qe=model.encode([q],normalize_embeddings=True)[0]
    scores=emb@qe; idx=np.argsort(-scores)[:k]; out=[]
    for rank,i in enumerate(idx,1):
        x=dict(chunks[int(i)]); x.update(score=float(scores[int(i)]),rank=rank,retriever='dense'); out.append(x)
    return out
