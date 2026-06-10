"""Run the MTEB BM25 baseline on the LIMIT retrieval tasks."""

import mteb

model = mteb.get_model("mteb/baseline-bm25s")

for task_name in ["LIMITSmallRetrieval", "LIMITRetrieval"]:
    tasks = mteb.get_tasks(tasks=[task_name])
    results = mteb.evaluate(model, tasks=tasks)
    for res in results:
        scores = res.only_main_score().to_dict()["scores"]
        print(f"\n=== {res.task_name} ===")
        full = res.to_dict()["scores"]["test"][0]
        for metric in [
            "main_score",
            "recall_at_2",
            "recall_at_10",
            "recall_at_100",
            "ndcg_at_10",
        ]:
            if metric in full:
                print(f"{metric}: {full[metric]}")
