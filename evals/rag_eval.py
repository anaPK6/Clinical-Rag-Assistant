"""Evaluation harness for the Clinical RAG Assistant.

Runs every gold Q&A pair through the pipeline and measures:

  - answer_correctness : for answerable Qs, does the answer contain the
                         expected keyword(s)?
  - citation_accuracy  : for answerable Qs, does a citation come from the
                         expected section?
  - refusal_correct    : for negative Qs, did the system refuse (no answer /
                         no citations)?  for answerable Qs, did it NOT refuse?
  - retrieval_recall   : was the expected section present in the retrieved
                         chunks at all (i.e. did retrieval surface it)?
  - latency_s          : wall-clock per question.

Metrics are logged to MLflow (local ./mlruns) and printed as a summary table.

Run:  python -m evals.rag_eval
"""
from __future__ import annotations

import time
from statistics import mean

from app.rag_pipeline import answer_question, retrieve
from evals.gold_set import GOLD


def _contains_all(text: str, keywords) -> bool:
    t = text.lower()
    return all(k.lower() in t for k in keywords)


def _is_refusal(result) -> bool:
    """A refusal = no citations survived AND the answer signals 'not found'."""
    if result.citations:
        return False
    ans = result.answer.lower()
    signals = ["cannot find", "can't find", "not find", "no mention",
               "not in the", "not provided", "unable to find", "does not"]
    return any(s in ans for s in signals) or len(ans.strip()) < 3


def evaluate(top_k: int = 5, use_relevance_gate: bool = False) -> dict:
    rows = []
    for item in GOLD:
        t0 = time.time()
        res = answer_question(item["question"], top_k=top_k, note_ids=[item["note_id"]],
                              use_relevance_gate=use_relevance_gate)
        latency = time.time() - t0

        refused = _is_refusal(res)
        # what sections did retrieval surface (for recall@k)?
        hits = retrieve(item["question"], top_k=top_k, note_ids=[item["note_id"]])
        retrieved_sections = {h.section for h in hits}
        cited_sections = {c.section for c in res.citations}

        if item["should_refuse"]:
            correctness = None
            citation_ok = None
            recall = None
            refusal_ok = refused
        else:
            correctness = _contains_all(res.answer, item["expected_keywords"])
            citation_ok = item["expected_section"] in cited_sections
            recall = item["expected_section"] in retrieved_sections
            refusal_ok = not refused  # should NOT refuse an answerable question

        rows.append(dict(
            note_id=item["note_id"], question=item["question"],
            should_refuse=item["should_refuse"], refused=refused,
            correctness=correctness, citation_ok=citation_ok,
            recall=recall, refusal_ok=refusal_ok, latency=latency,
        ))
        mark = "REFUSE" if refused else "ANSWER"
        flag = "ok" if refusal_ok else "MISS"
        print(f"[{flag:4s}] {mark} ({latency:4.1f}s)  {item['question'][:55]}")

    # aggregate
    answerable = [r for r in rows if not r["should_refuse"]]
    negatives = [r for r in rows if r["should_refuse"]]

    def rate(items, key):
        vals = [r[key] for r in items if r[key] is not None]
        return mean(vals) if vals else 0.0

    summary = {
        "n_total": len(rows),
        "n_answerable": len(answerable),
        "n_refusal_cases": len(negatives),
        "answer_correctness": rate(answerable, "correctness"),
        "citation_accuracy": rate(answerable, "citation_ok"),
        "retrieval_recall_at_k": rate(answerable, "recall"),
        "refusal_rate_on_negatives": rate(negatives, "refusal_ok"),
        "false_refusal_rate": 1 - rate(answerable, "refusal_ok"),
        "latency_mean_s": mean(r["latency"] for r in rows),
        "latency_max_s": max(r["latency"] for r in rows),
    }
    return {"rows": rows, "summary": summary}


def main():
    import sys
    gate = "--gate" in sys.argv
    print(f"Running RAG evaluation on the gold set… (relevance_gate={'ON' if gate else 'OFF'})\n")
    out = evaluate(use_relevance_gate=gate)
    s = out["summary"]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in s.items():
        if isinstance(v, float):
            print(f"  {k:28s}: {v:.2%}" if "rate" in k or "accuracy" in k
                  or "correctness" in k or "recall" in k else f"  {k:28s}: {v:.2f}")
        else:
            print(f"  {k:28s}: {v}")

    # MLflow logging (best-effort; don't fail the eval if MLflow hiccups)
    try:
        import mlflow
        mlflow.set_experiment("clinical-rag-eval")
        with mlflow.start_run():
            mlflow.log_metrics({k: v for k, v in s.items() if isinstance(v, (int, float))})
            mlflow.log_param("n_gold", s["n_total"])
        print("\nLogged to MLflow (./mlruns). View with:  mlflow ui")
    except Exception as e:
        print(f"\n[MLflow logging skipped: {e}]")


if __name__ == "__main__":
    main()
