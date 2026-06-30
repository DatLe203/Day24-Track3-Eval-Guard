from __future__ import annotations

"""Module 4: RAGAS Evaluation - 4 metrics + failure analysis."""

import os, sys, json, re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _overlap_score(source: str, target: str) -> float:
    source_tokens = _tokens(source)
    target_tokens = _tokens(target)
    if not source_tokens or not target_tokens:
        return 0.0
    return len(source_tokens & target_tokens) / len(target_tokens)


def _fallback_evaluate(questions: list[str], answers: list[str],
                       contexts: list[list[str]], ground_truths: list[str]) -> dict:
    per_question: list[EvalResult] = []
    for question, answer, ctxs, ground_truth in zip(questions, answers, contexts, ground_truths):
        context_text = "\n".join(ctxs)
        faithfulness = _overlap_score(context_text, answer)
        answer_relevancy = _overlap_score(question + " " + ground_truth, answer)
        context_precision = sum(_overlap_score(ground_truth, ctx) for ctx in ctxs) / max(len(ctxs), 1)
        context_recall = _overlap_score(context_text, ground_truth)
        per_question.append(EvalResult(
            question=question,
            answer=answer,
            contexts=ctxs,
            ground_truth=ground_truth,
            faithfulness=round(faithfulness, 4),
            answer_relevancy=round(answer_relevancy, 4),
            context_precision=round(context_precision, 4),
            context_recall=round(context_recall, 4),
        ))

    def avg(metric: str) -> float:
        if not per_question:
            return 0.0
        return round(sum(getattr(item, metric) for item in per_question) / len(per_question), 4)

    return {
        "faithfulness": avg("faithfulness"),
        "answer_relevancy": avg("answer_relevancy"),
        "context_precision": avg("context_precision"),
        "context_recall": avg("context_recall"),
        "per_question": per_question,
    }


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation, falling back to transparent lexical estimates."""
    if os.getenv("LAB18_OFFLINE") == "1":
        return _fallback_evaluate(questions, answers, contexts, ground_truths)

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=list(row["contexts"]),
                ground_truth=row["ground_truth"],
                faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                context_precision=float(row.get("context_precision", 0.0) or 0.0),
                context_recall=float(row.get("context_recall", 0.0) or 0.0),
            )
            for _, row in df.iterrows()
        ]

        def metric_value(name: str) -> float:
            if name in result:
                return float(result[name])
            if not per_question:
                return 0.0
            return float(sum(getattr(item, name) for item in per_question) / len(per_question))

        return {
            "faithfulness": metric_value("faithfulness"),
            "answer_relevancy": metric_value("answer_relevancy"),
            "context_precision": metric_value("context_precision"),
            "context_recall": metric_value("context_recall"),
            "per_question": per_question,
        }
    except Exception as exc:
        print(f"  Warning: RAGAS evaluation failed, using lexical fallback: {exc}")
        return _fallback_evaluate(questions, answers, contexts, ground_truths)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using a Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating or answer not grounded in retrieved context",
                         "Tighten prompt, lower temperature, and cite only retrieved evidence"),
        "context_recall": ("Missing relevant chunks",
                           "Improve chunking, add BM25 terms, or enrich chunks with HyQA/context"),
        "context_precision": ("Too many irrelevant chunks",
                              "Add stronger reranking, metadata filters, or reduce top-k"),
        "answer_relevancy": ("Answer does not directly match the question",
                             "Improve answer prompt and preserve question intent during generation"),
    }
    rows = []
    for result in eval_results:
        metrics = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        avg_score = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        rows.append({
            "question": result.question,
            "answer": result.answer,
            "ground_truth": result.ground_truth,
            "avg_score": round(avg_score, 4),
            "worst_metric": worst_metric,
            "score": round(metrics[worst_metric], 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
            "error_tree": f"{worst_metric} -> {diagnosis} -> {suggested_fix}",
        })
    return sorted(rows, key=lambda item: item["avg_score"])[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON."""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")


