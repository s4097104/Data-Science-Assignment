# Protocol and provenance

40 questions authored before evaluation execution: 12 direct, 8 paraphrase, 8 comparison, 4 ambiguous, 8 unsupported. Retrieval denominators use only the 28 answerable questions. Labels are AI-assisted and source-reviewed, not independent human judgments.

This is a retrospective diagnostic benchmark, not a blind held-out test: source content and original 18 questions had already been inspected. Related questions are correlated. No significance claim or independent-generalisation claim is made. Do not rename this set held-out. Future benchmark splits must separate topic groups.

All retrievers use the same ten passages, query strings and judgments. BM25 uses k1=1.5 and b=.75. The length-normalisation ablation changes only b to 0. TF-IDF uses word unigrams+bigrams and cosine similarity. Lexical RRF combines BM25 and TF-IDF top-five lists with equal weights and constant 60; it is not dense/hybrid semantic retrieval. Top k=1,3,5 are recorded.

A zero-overlap screen abstains only when the maximum raw retrieval score is zero. This is a diagnostic test of a retrieval gate, not a test of LLM refusal. Eight unsupported questions and four ambiguous prompts have reference behaviour but no generated responses.

Reference arithmetic is checked in code. Source numerical values are inspected in the original PDF. Independent human adjudication, generated-answer scoring and a user study remain outstanding. No participant measurements or personal reflections were invented.

Dense comparison was subsequently executed with the same frozen questions and judgments, using the prototype model via ONNX CPU, mean pooling, L2 normalisation and 512-token truncation. Model hashes and token lengths are retained. Semantic hybrid combines BM25 and dense top-five lists, weights .5 each, constant 60. Ties use original corpus order. No thresholds were tuned against these results.
