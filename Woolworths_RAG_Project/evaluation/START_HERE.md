# Strengthened report and evaluation package

Upload this ZIP to Overleaf; compile main.tex with pdfLaTeX. The PDF is a completed revision of the written analysis, with real new retrieval experiments. It does not claim completed user studies or generated-answer evaluation.

For final assessment, use the official course template and add its clickable project URL. AI declaration and contribution form remain for separate completion, as requested. Each member must supply their own reflection and actual teamwork evidence.

## Main new evidence
- 40 source-reviewed questions: 28 answerable, four ambiguous and eight unsupported.
- Actual BM25, dense MiniLM ONNX and semantic hybrid comparison on one corpus.
- TF-IDF and lexical fusion controls; cut-off sensitivity at k=1,3,5.
- Executed BM25 length-normalisation ablation.
- Unsupported-question zero-overlap test; measured tokenizer lengths.
- Reference-answer arithmetic checks, full rankings, metrics and hashes.

Read BENCHMARK_PROTOCOL.md. Labels are AI-assisted and not independently human-adjudicated. This is a diagnostic set, not an independent held-out sample. Correlated paraphrases must not be used to overstate statistical evidence.

## Run retrieval evaluation
Install requirements.txt in a suitable Python environment, then:

    python evaluate.py

Download these files from the official model repository, placing them in one model directory:
https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1

- onnx/model.onnx (save as model.onnx)
- tokenizer.json
- config.json

Then run:

    python evaluate_dense.py /path/to/model_directory

The report's run uses mean pooling, L2 normalisation and 512-token truncation. Exact file hashes are in new_evidence/dense_manifest.json. If the remote files change, your hashes/results may differ. Model weights are not included in this ZIP.

## Complete answer-generation evaluation on your machine
Start Ollama with your chosen model available. Then:

    python run_answers.py --model llama3.2:3b

This runs 40 questions for each of three retrieval methods, with the same prompt and five contexts. It can take substantial time. It writes genuine outputs or explicit errors to generation_outputs.jsonl. It does not award correctness automatically. It uses a 16,384-token context request; confirm model support and inspect the recorded runtime token counts for truncation.

Two team members should independently score responses using answer_scoring_template.csv, then reconcile. Blank values mean unmeasured. An execution error is not an abstention. Report answerable correctness on 28 questions, correct refusal on eight unsupported questions, and ambiguity handling on four. Calculate claim faithfulness and citation support separately; do not merely search for matching numbers. Record model digest/version and machine details alongside actual outputs.

## Blocking evidence issues
- The GitHub main archive inspected on 25 October 2026 contained a customer-churn project rather than RAG artifacts. Upload the correct code, data and evidence or replace the report link with the correct repository.
- Vijay's larger-corpus draft lacks raw output logs and referenced screenshots. Its scores remain historical, not verified evidence for this benchmark.
- The report cannot invent mentor meetings, commits, team reflections or participant time savings. Add authentic evidence before submission.
