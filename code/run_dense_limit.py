"""Run dense embedding models on the LIMIT retrieval tasks for comparison with BM25."""

import time
import traceback

import mteb

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
    "Qwen/Qwen3-Embedding-0.6B",
]

TASKS = ["LIMITSmallRetrieval", "LIMITRetrieval"]
METRICS = ["recall_at_2", "recall_at_10", "recall_at_100", "ndcg_at_10"]

for model_name in MODELS:
    for task_name in TASKS:
        try:
            start = time.time()
            model = mteb.get_model(model_name)
            tasks = mteb.get_tasks(tasks=[task_name])
            results = mteb.evaluate(model, tasks=tasks)
            scores = results[0].to_dict()["scores"]["test"][0]
            line = " ".join(f"{m}={scores[m]:.4f}" for m in METRICS if m in scores)
            print(
                f"RESULT {model_name} {task_name} {line} "
                f"({time.time() - start:.0f}s)",
                flush=True,
            )
        except Exception:
            print(f"FAILED {model_name} {task_name}", flush=True)
            traceback.print_exc()
        finally:
            del model
