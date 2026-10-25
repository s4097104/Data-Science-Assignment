import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from config import CHUNKS_FILE
from load_chunks import load_chunks
from dense_retriever import build_embeddings
chunks=load_chunks(CHUNKS_FILE); path=build_embeddings(chunks); print(f'Built embeddings for {len(chunks)} chunks: {path}')
