# User-Friendly Website Guide

## 1. Install dependencies

From the project folder:

```bash
pip install -r requirements.txt
```

## 2. Build the semantic-search index

If you have not already done so:

```bash
python src/build_embeddings.py
```

This creates `outputs/dense_embeddings.npy`.

## 3. Start Ollama

Install Ollama, then pull the model used by the project:

```bash
ollama pull llama3.2
```

Make sure Ollama is running before asking questions.

## 4. Start the website

```bash
streamlit run app.py
```

Streamlit will display a local address, normally:

```text
http://localhost:8501
```

Open that address in your browser.

## What the website provides

- Clean dashboard-style landing page
- Natural-language question box
- Example questions in the sidebar
- BM25 + dense hybrid retrieval
- Generated answer from Ollama
- Supporting source passages
- Page/chunk references
- Retrieval-detail view for BM25 and dense search
- Clear error messages when embeddings or Ollama are unavailable
- Adjustable number of retrieved sources

## Important project limitation

The current supplied knowledge base is based on the FY25 Five Year Summary / current project dataset. It should not be described as the complete FY2025 Annual Report unless the full Annual Report has been processed into the knowledge base.
