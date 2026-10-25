import pandas as pd
from pathlib import Path

def load_chunks(path):
    df=pd.read_csv(path)
    chunks=[]
    for _,r in df.iterrows():
        chunks.append({'chunk_id':str(r['chunk_id']),'page':int(r['page']),'source':str(r.get('source','Woolworths FY25')),'text':str(r['text'])})
    return chunks
