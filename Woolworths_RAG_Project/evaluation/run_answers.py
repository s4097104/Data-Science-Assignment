"""Generate evaluation responses on a machine running Ollama.
python run_answers.py --model llama3.2:3b
Requires evaluate.py and evaluate_dense.py outputs. No correctness is auto-awarded.
Outputs are actual model responses; failure is logged separately from abstention.
"""
import argparse,csv,json,time,urllib.request,urllib.error
from pathlib import Path
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--model',default='llama3.2:3b');ap.add_argument('--url',default='http://localhost:11434/api/generate');a=ap.parse_args()
qs=json.loads((P/'benchmark.json').read_text());cs={c['chunk_id']:c for c in csv.DictReader(open(P/'audit_inputs/data/chunks.csv',encoding='utf-8-sig'))}
rs=list(csv.DictReader(open(P/'new_evidence/rankings.csv')))+list(csv.DictReader(open(P/'new_evidence/dense_rankings.csv')))
out=P/'generation_outputs.jsonl'
with open(out,'w') as f:
 for q in qs:
  for method in ['BM25','Dense_ONNX','Hybrid_BM25_Dense']:
   rr=sorted([r for r in rs if r['question_id']==q['id'] and r['method']==method],key=lambda r:int(r['rank']))[:5]
   assert len(rr)==5
   context='\n\n'.join(f"[chunk {r['chunk_id']} | page {r['page']}]\n{cs[r['chunk_id']]['text']}" for r in rr)
   system='Answer only using supplied financial-report context. Preserve year, units and accounting basis. Cite supporting chunk IDs and pages. If evidence is insufficient, say so. Clarify ambiguous requests or distinguish the relevant measures. Do not invent values.'
   payload=dict(model=a.model,system=system,prompt=f"Question: {q['question']}\n\nContext:\n{context}",stream=False,options={'temperature':0,'num_ctx':16384})
   row=dict(question_id=q['id'],method=method,model=a.model,context_ids=[r['chunk_id'] for r in rr],request=payload)
   t=time.perf_counter()
   try:
    req=urllib.request.Request(a.url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=300) as response:result=json.load(response)
    row.update(status='ok',answer=result.get('response',''),runtime_response=result)
   except Exception as e:row.update(status='error',error=str(e))
   row['elapsed_seconds']=time.perf_counter()-t;f.write(json.dumps(row)+'\n');f.flush()
print(out)
