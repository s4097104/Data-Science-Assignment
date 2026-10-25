# Woolworths FY25 Report Q&A — RAG Project

## Goal
Build a Retrieval-Augmented Generation (RAG) question-answering system for the Woolworths Group FY2025 Annual Report.

Pipeline:
PDF → chunks + metadata → BM25 → dense retrieval → hybrid retrieval → Ollama LLM → answer + sources → evaluation → UI

## Important data limitation
The included PDF is the **9-page FY25 Five Year Summary**, not the complete 150+ page FY2025 Annual Report named in the team's Milestone 1 report. This package is therefore a working starter/development pipeline. Do not claim final performance on the full annual report until the complete report is processed.

## Install
Windows:
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run BM25
```bash
python run_bm25.py
```

## Build embeddings
```bash
python src/build_embeddings.py
```

## Dense retrieval
```bash
python run_dense.py
```

## Hybrid retrieval
```bash
python run_hybrid.py
```

## Ollama + RAG
Install Ollama separately, pull a model your computer can run, then:
```bash
ollama pull llama3.2
python run_rag.py
```

## UI
```bash
streamlit run app.py
```

## Beginner meaning
- Chunk = small piece of the report.
- Metadata = page/source/section information attached to a chunk.
- BM25 = keyword/lexical search.
- Embedding = numerical representation of meaning.
- Dense retrieval = semantic search using embeddings.
- Hybrid = combine BM25 and dense rankings.
- RAG = retrieve evidence first, then give it to an LLM to generate an answer.
- Ollama = local runtime/API for the LLM.
- Source attribution = show where the answer came from.
- Evaluation = measure whether retrieval and answers are actually good.
