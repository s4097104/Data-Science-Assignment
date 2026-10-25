import sys,pandas as pd
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from config import CHUNKS_FILE,EVAL_FILE,OUTPUT_DIR
from load_chunks import load_chunks
from bm25_retriever import bm25_search
chunks=load_chunks(CHUNKS_FILE); qs=pd.read_csv(EVAL_FILE); rows=[]
for _,q in qs.iterrows():
    for x in bm25_search(chunks,q['question'],5): rows.append({'question_id':q['question_id'],'question':q['question'],'gold_page':q['expected_source_page'],'rank':x['rank'],'chunk_id':x['chunk_id'],'retrieved_page':x['page'],'score':x['score'],'text_preview':x['text'][:500]})
out=pd.DataFrame(rows); OUTPUT_DIR.mkdir(exist_ok=True); out.to_csv(OUTPUT_DIR/'bm25_results.csv',index=False); print(out.to_string(index=False)); print('Saved',OUTPUT_DIR/'bm25_results.csv')
