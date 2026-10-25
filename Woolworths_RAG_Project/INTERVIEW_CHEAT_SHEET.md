# Interview Cheat Sheet

## 30-second project answer
"We built a Retrieval-Augmented Generation system that lets investors and analysts ask natural-language questions about Woolworths Group's FY2025 Annual Report. The system retrieves relevant report passages using BM25 and dense embeddings, combines them using hybrid retrieval, and passes the evidence to a local LLM through Ollama to generate a grounded answer with source pages. We then evaluate retrieval and answer quality."

## Why RAG?
A normal LLM may not have the exact document, may be outdated, or may hallucinate. RAG retrieves evidence from the target document first and uses that evidence as context.

## BM25
Lexical/keyword retrieval. Strong when query words overlap with the source.

## Dense embeddings
Convert text into vectors representing semantic meaning. Useful when wording differs but meaning is similar.

## Hybrid retrieval
Combines lexical and semantic retrieval because they have complementary strengths.

## Ollama
Runs the selected local LLM and exposes a local API. The LLM generates the final answer from retrieved context.

## Chunking
Splits a long report into smaller searchable pieces while preserving metadata such as page and section.

## Source attribution
Shows the page/chunk supporting the answer so users can verify it.

## Biggest technical risk
Retrieval failure. If the correct evidence is not retrieved, the LLM may not have enough information to answer correctly.

## Evaluation
Retrieval: NDCG, Recall, Precision, MRR.
Generation: correctness, relevance, faithfulness, source attribution.

## Your Milestone 1 role
Retrieval & pipeline engineering — core RAG code.

## Existing team result
BM25 NDCG@5 = 0.476 for questions close to source wording and 0.193 for inference questions. These are the reported baseline results from Milestone 1; do not present them as dense/hybrid results.

## Critical honesty point
The current starter package uses the uploaded 9-page Five Year Summary. The final project should use the complete FY2025 Annual Report specified in Milestone 1 before claiming final project performance.
