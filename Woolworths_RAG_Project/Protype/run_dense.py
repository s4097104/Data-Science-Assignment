import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from config import CHUNKS_FILE
from load_chunks import load_chunks
from dense_retriever import dense_search
chunks=load_chunks(CHUNKS_FILE); q=input('Enter a question: '); [print(f"\nRank {x['rank']} | score={x['score']:.4f} | page={x['page']} | {x['chunk_id']}\n{x['text'][:800]}") for x in dense_search(chunks,q,5)]
