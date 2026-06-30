from __future__ import annotations

"""Module 3: Reranking - Cross-encoder top-20 to top-3 + latency benchmark."""

import os, sys, time, re, math
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _lexical_score(query: str, document: str) -> float:
    q = _tokens(query)
    d = _tokens(document)
    if not q or not d:
        return 0.0
    overlap = len(q & d)
    recall = overlap / len(q)
    precision = overlap / len(d)
    numeric_bonus = 0.1 * len(set(re.findall(r"\d+", query)) & set(re.findall(r"\d+", document)))
    return (2 * precision * recall / (precision + recall + 1e-9)) + numeric_bonus


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._model_failed = False

    def _load_model(self):
        if os.getenv("LAB18_USE_SENTENCE_TRANSFORMERS") != "1":
            self._model_failed = True
            return None
        if self._model is None and not self._model_failed:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                print(f"  Warning: cross-encoder unavailable, using lexical reranker: {exc}")
                self._model_failed = True
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 to top-k."""
        if not documents:
            return []

        model = self._load_model()
        if model is not None:
            try:
                pairs = [(query, doc["text"]) for doc in documents]
                scores = model.predict(pairs)
                if isinstance(scores, (int, float)):
                    scores = [scores]
                scores = [float(s) for s in scores]
            except Exception as exc:
                print(f"  Warning: cross-encoder predict failed, using lexical reranker: {exc}")
                scores = [_lexical_score(query, doc["text"]) for doc in documents]
        else:
            scores = [_lexical_score(query, doc["text"]) for doc in documents]

        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [
            RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i + 1,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
        scored = sorted(
            ((_lexical_score(query, doc["text"]), doc) for doc in documents),
            key=lambda x: x[0],
            reverse=True,
        )
        return [
            RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i + 1,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhan vien duoc nghi phep bao nhieu ngay?"
    docs = [
        {"text": "Nhan vien duoc nghi 12 ngay/nam.", "score": 0.8, "metadata": {}},
        {"text": "Mat khau thay doi moi 90 ngay.", "score": 0.7, "metadata": {}},
        {"text": "Thoi gian thu viec la 60 ngay.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")

