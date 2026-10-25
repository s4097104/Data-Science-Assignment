import requests
from config import OLLAMA_URL,OLLAMA_MODEL
SYSTEM_PROMPT="""You are a financial-report QA assistant. Answer only from the supplied report context. If the context is insufficient, say so. Do not invent figures. Mention source pages."""
def generate_answer(question,contexts,model=OLLAMA_MODEL):
    ctx='\n\n'.join(f"[SOURCE {i} | page {x['page']} | {x['chunk_id']}]\n{x['text']}" for i,x in enumerate(contexts,1))
    prompt=f"Question:\n{question}\n\nReport context:\n{ctx}\n\nAnswer directly using only the context and mention the relevant source page(s)."
    payload={'model':model,'system':SYSTEM_PROMPT,'prompt':prompt,'stream':False,'options':{'temperature':0}}
    r=requests.post(OLLAMA_URL,json=payload,timeout=180); r.raise_for_status(); return r.json().get('response','').strip()
