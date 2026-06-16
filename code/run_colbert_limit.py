"""Run ColBERT v2 (multi-vector late interaction) on the LIMIT retrieval tasks.

ColBERT is a multi-vector model, the paradigm the LIMIT paper argues should
escape the single-vector dimensionality bound. Retrieval uses pylate's PLAID
index with MaxSim scoring.
"""

import time
import traceback

import mteb

MODEL = "colbert-ir/colbertv2.0"
TASKS = ["LIMITSmallRetrieval", "LIMITRetrieval"]
METRICS = ["recall_at_2", "recall_at_10", "recall_at_100", "ndcg_at_10"]

for task_name in TASKS:
    try:
        start = time.time()
        model = mteb.get_model(MODEL)
        tasks = mteb.get_tasks(tasks=[task_name])
        results = mteb.evaluate(model, tasks=tasks)
        scores = results[0].to_dict()["scores"]["test"][0]
        line = " ".join(f"{m}={scores[m]:.4f}" for m in METRICS if m in scores)
        print(f"RESULT {MODEL} {task_name} {line} ({time.time() - start:.0f}s)", flush=True)
    except Exception:
        print(f"FAILED {MODEL} {task_name}", flush=True)
        traceback.print_exc()
