import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from rag import RAGPipeline
p=RAGPipeline(); q=input('Ask a Woolworths question: '); r=p.answer(q); print('\nANSWER\n'+r['answer']); print('\nSOURCES'); [print(f"- Page {x['page']} | {x['chunk_id']} | score={x['score']:.4f}") for x in r['hybrid']]
