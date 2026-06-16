# LIMIT Benchmark Results

Reproduction of the LIMIT benchmark across three retrieval paradigms — lexical
(BM25), single-vector dense embeddings, and multi-vector late interaction
(ColBERT) — using the [MTEB](https://github.com/embeddings-benchmark/mteb)
framework (`mteb==2.15.1`).

All numbers are percentages. Metrics are from MTEB's standard retrieval output.
Runner scripts: [`run_bm25_limit.py`](./run_bm25_limit.py),
[`run_dense_limit.py`](./run_dense_limit.py),
[`run_colbert_limit.py`](./run_colbert_limit.py).

## LIMIT (full, 50k documents)

| Model | Paradigm | recall@2 | recall@10 | recall@100 | nDCG@10 |
|---|---|---|---|---|---|
| `mteb/baseline-bm25s` | lexical (sparse) | **85.7** | **90.3** | **93.5** | **88.4** |
| `colbert-ir/colbertv2.0` | multi-vector | 25.7 | 37.6 | 52.3 | 32.4 |
| `BAAI/bge-base-en-v1.5` | single-vector | 0.7 | 1.5 | 4.5 | 1.1 |
| `Qwen/Qwen3-Embedding-0.6B` | single-vector | 1.1 | 1.9 | 3.9 | 1.5 |
| `intfloat/e5-base-v2` | single-vector | 0.4 | 1.1 | 3.8 | 0.8 |
| `sentence-transformers/all-MiniLM-L6-v2` | single-vector | 0.1 | 0.6 | 2.9 | 0.4 |
| `sentence-transformers/all-mpnet-base-v2` | single-vector | 0.0 | 0.1 | 0.8 | 0.0 |

## LIMIT-small (46 documents)

On the small corpus recall@100 is trivially 100% for every model (only 46 docs),
so **recall@2** is the meaningful signal.

| Model | Paradigm | recall@2 | recall@10 | recall@100 | nDCG@10 |
|---|---|---|---|---|---|
| `mteb/baseline-bm25s` | lexical (sparse) | **97.8** | 100.0 | 100.0 | **99.2** |
| `colbert-ir/colbertv2.0` | multi-vector | 97.0 | 100.0 | 100.0 | 98.8 |
| `intfloat/e5-base-v2` | single-vector | 18.3 | 50.4 | 100.0 | 34.0 |
| `BAAI/bge-base-en-v1.5` | single-vector | 17.2 | 49.2 | 100.0 | 32.9 |
| `sentence-transformers/all-MiniLM-L6-v2` | single-vector | 16.2 | 48.0 | 100.0 | 31.4 |
| `Qwen/Qwen3-Embedding-0.6B` | single-vector | 14.6 | 40.7 | 100.0 | 27.2 |
| `sentence-transformers/all-mpnet-base-v2` | single-vector | 7.8 | 32.9 | 100.0 | 19.4 |

## Takeaways

These results reproduce the paper's central claim:

- **Single-vector dense embedders collapse on full LIMIT** (all below 5%
  recall@100) despite the task being lexically trivial ("Who likes X?" →
  "... likes X ..."). The bottleneck is the single-vector dimensionality bound,
  not task difficulty.
- **Scale and recency do not help.** The 2025 600M-param Qwen3-Embedding-0.6B
  performs no better than the 22M-param all-MiniLM-L6-v2 — even on the 46-doc
  small set, no single-vector model reliably picks the 2 relevant documents.
- **BM25 nearly solves it** (93.5% recall@100). A lexical model behaves like a
  very high-dimensional sparse single-vector model, so the dimensionality bound
  never binds.
- **Multi-vector late interaction (ColBERT v2) escapes the bound.** It matches
  BM25 on LIMIT-small (97.0% recall@2) and is an order of magnitude better than
  single-vector models on full LIMIT (52.3% vs <5% recall@100), consistent with
  the paper's argument that the limitation is specific to the single-vector
  paradigm. It still trails BM25 on the full corpus, partly due to the
  approximate PLAID index used for retrieval.

## Hardware / reproducibility

Run on an Apple M4 (10-core, 16 GB RAM) using the Metal (MPS) backend for the
neural models. Approximate full-corpus (50k) wall-clock times:

| Model | Full-corpus time |
|---|---|
| `mteb/baseline-bm25s` | ~17 s (CPU) |
| `all-MiniLM-L6-v2` (22M) | ~2.5 min |
| base dense models (110M) | ~12 min each |
| `Qwen/Qwen3-Embedding-0.6B` | ~84 min |
| `colbert-ir/colbertv2.0` | ~40 min (encode + PLAID index) |

Raw per-model JSON is written by MTEB to `~/.cache/mteb/results/`.

7B-class leaderboard embedders were not run — they exceed this machine's memory
budget; the paper shows even those models stay below ~20% recall@100 on full
LIMIT.
