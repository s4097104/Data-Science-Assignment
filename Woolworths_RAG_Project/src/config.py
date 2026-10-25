from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA_DIR=ROOT/'data'; OUTPUT_DIR=ROOT/'outputs'
CHUNKS_FILE=DATA_DIR/'chunks.csv'
EVAL_FILE=DATA_DIR/'evaluation_questions.csv'
TOP_K=5
EMBEDDING_MODEL='sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
OLLAMA_URL='http://localhost:11434/api/generate'
OLLAMA_MODEL='llama3.2'
RRF_K=60
BM25_WEIGHT=0.5
DENSE_WEIGHT=0.5
