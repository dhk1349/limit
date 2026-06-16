# On the Theoretical Limitations of Embedding-based Retrieval

This repository contains the official resources for the paper "[On the Theoretical Limitations of Embedding-based Retrieval](https://arxiv.org/abs/2508.21038)".
This work introduces the **LIMIT** dataset,
designed to stress-test embedding models based on theoretical principles.
We show that for any given embedding dimension `d`,
there exists a combination of documents that cannot be returned by any query.
We use this theory to instantiate the dataset LIMIT,
finding that even state-of-the-art models struggle: highlighting a fundamental
limitation of the current single-vector embedding paradigm.

![LIMIT Dataset Concept](assets/LIMIT.png)

## Overview

* [Data](#data)
* [Code](#code)
* [Evaluation](#evaluation)
* [Benchmark Results](#benchmark-results)
* [Citation](#citation)
* [License and disclaimer](#license-and-disclaimer)

## Data

The datasets used in our experiments are available in the `data/` directory of this repository, formatted in [MTEB](https://github.com/embeddings-benchmark/mteb) style (i.e. json lines).

Each dataset contains:

- A `queries.json` file containing a line for each of the 1000 queries, each with an `_id` and the `text` field.
- A `corpus.json` file containing a line for each of the 50k (or 46 if using the `small` version) documents, each with an `_id`, `text` and empty `title` field.
- A `qrels.json` file containing rows for each of the 2000 relevant query->doc mappings, mapping `query-id` of the queries into the `corpus-id` in the documents, with `score` indicating relevance.

* **Full Dataset (`limit`):** The complete dataset, containing 50k documents.
  * [Link to `data/limit`](./data/limit)

* **Small Sample (`limit-small`):** A smaller version with only the 46 documents relevant to the queries.
  * [Link to `data/limit-small`](./data/limit-small)

## Code

We provide code to generate the LIMIT style datasets,
as well as to run the free embedding experiment in the `code/` folder.

* **Dataset Generation:** To generate the dataset from scratch, you can use the Jupyter notebook located at `code/generate_limit_dataset.ipynb`. This contains all necessary steps and dependencies.
  * [Link to `code/generate_limit_dataset.ipynb`](./code/generate_limit_dataset.ipynb)

* **Free Embedding Experiments:** The script to run the free embedding experiments can be found in `code/free_embedding_experiment.py`.
  * [Link to `code/free_embedding_experiment.py`](./code/free_embedding_experiment.py)

If you use the free embedding code,
you'll need to install the following requirements.

### Installation

We recommend using the [`uv` package manager](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Create a virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r https://raw.githubusercontent.com/google-deepmind/limit/refs/heads/main/code/requirements.txt
```

## Loading with Huggingface Datasets
You can also load the data using the `datasets` library from Huggingface ([LIMIT](https://huggingface.co/datasets/orionweller/LIMIT), [LIMIT-small](https://huggingface.co/datasets/orionweller/LIMIT-small)):
```python
from datasets import load_dataset
ds = load_dataset("orionweller/LIMIT-small", "corpus") # also available: queries, test (contains qrels).
```

## Evaluation

Evaluation was done using the [MTEB framework](https://github.com/embeddings-benchmark/mteb). You can reproduce this **only on the [v2.0.0 branch](https://github.com/embeddings-benchmark/mteb/tree/w_limit)** (soon to be `main`). Note that the v2.0.0 branch is changing rapidly, so **please install the version pinned in the requirements** until it becomes main. An example is:

```python
import mteb
from sentence_transformers import SentenceTransformer

# load the model using MTEB
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = mteb.get_model(model_name) # will default to SentenceTransformers(model_name) if not implemented in MTEB
# or using SentenceTransformers
model = SentenceTransformer(model_name)

# select the desired tasks and evaluate
tasks = mteb.get_tasks(tasks=["LIMITSmallRetrieval"]) # or use LIMITRetrieval for the full dataset
results = mteb.evaluate(model, tasks=tasks)
```

Please see their Github for more details.

## Benchmark Results

We reproduced the benchmark on this fork across all three retrieval paradigms — lexical (BM25), single-vector dense embeddings, and multi-vector late interaction (ColBERT v2) — using MTEB (`mteb==2.15.1`). Runner scripts are in [`code/`](./code) (`run_bm25_limit.py`, `run_dense_limit.py`, `run_colbert_limit.py`) and the full breakdown is in [`code/RESULTS.md`](./code/RESULTS.md).

**Full LIMIT (50k documents), as percentages:**

| Model | Paradigm | recall@2 | recall@10 | recall@100 | nDCG@10 |
|---|---|---|---|---|---|
| `mteb/baseline-bm25s` | lexical (sparse) | **85.7** | **90.3** | **93.5** | **88.4** |
| `colbert-ir/colbertv2.0` | multi-vector | 25.7 | 37.6 | 52.3 | 32.4 |
| `BAAI/bge-base-en-v1.5` | single-vector | 0.7 | 1.5 | 4.5 | 1.1 |
| `Qwen/Qwen3-Embedding-0.6B` | single-vector | 1.1 | 1.9 | 3.9 | 1.5 |
| `intfloat/e5-base-v2` | single-vector | 0.4 | 1.1 | 3.8 | 0.8 |
| `sentence-transformers/all-MiniLM-L6-v2` | single-vector | 0.1 | 0.6 | 2.9 | 0.4 |
| `sentence-transformers/all-mpnet-base-v2` | single-vector | 0.0 | 0.1 | 0.8 | 0.0 |

These results reproduce the paper's central claim: single-vector dense embedders collapse on full LIMIT (all below 5% recall@100) despite the lexically trivial task, while BM25 nearly solves it and multi-vector ColBERT v2 escapes the single-vector dimensionality bound (matching BM25 on LIMIT-small at 97% recall@2). See [`code/RESULTS.md`](./code/RESULTS.md) for the LIMIT-small table, hardware notes, and timings.

## Citation

If you use this work, please cite the paper as:

```bibtex
@article{weller2025theoretical,
  title={On the Theoretical Limitations of Embedding-Based Retrieval},
  author={Weller, Orion and Boratko, Michael and Naim, Iftekhar and Lee, Jinhyuk},
  journal={arXiv preprint arXiv:2508.21038},
  year={2025}
}
```

## License and disclaimer

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
