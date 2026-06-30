from __future__ import annotations

"""Module 2: Hybrid Search - BM25 (Vietnamese) + Dense + RRF."""

import os, sys, re, math, hashlib
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into searchable whitespace tokens."""
    try:
        from underthesea import word_tokenize

        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception:
        return re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in segment_vietnamese(text).split() if t.strip()]


class _SimpleBM25:
    """Small BM25 fallback for environments where rank_bm25 is unavailable."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(doc) for doc in corpus_tokens) / max(len(corpus_tokens), 1)
        self.doc_freq: dict[str, int] = {}
        for doc in corpus_tokens:
            for token in set(doc):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        n_docs = max(len(self.corpus_tokens), 1)
        scores: list[float] = []
        for doc in self.corpus_tokens:
            doc_len = len(doc) or 1
            score = 0.0
            for token in query_tokens:
                tf = doc.count(token)
                if tf == 0:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
                score += idf * (tf * (self.k1 + 1)) / denom
            scores.append(score)
        return scores


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [_tokens(chunk["text"]) for chunk in chunks]
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            self.bm25 = _SimpleBM25(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []
        query_tokens = _tokens(query)
        scores = list(self.bm25.get_scores(query_tokens))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            doc = self.documents[idx]
            results.append(SearchResult(
                text=doc["text"],
                score=float(scores[idx]),
                metadata=doc.get("metadata", {}),
                method="bm25",
            ))
        return results


class DenseSearch:
    def __init__(self):
        self.client = None
        self._encoder = None
        self._documents: list[dict] = []
        self._vectors: list[list[float]] = []
        try:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        except Exception:
            self.client = None

    def _hash_embed(self, text: str, dim: int = 384) -> list[float]:
        vector = [0.0] * dim
        for token in _tokens(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def _get_encoder(self):
        if os.getenv("LAB18_USE_SENTENCE_TRANSFORMERS") != "1":
            raise RuntimeError("SentenceTransformer dense model disabled; set LAB18_USE_SENTENCE_TRANSFORMERS=1 to enable")
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant, falling back to in-memory dense vectors."""
        self._documents = chunks
        texts = [c["text"] for c in chunks]
        try:
            vectors = self._get_encoder().encode(texts, show_progress_bar=False)
            vector_lists = [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]
        except Exception as exc:
            print(f"  Warning: dense model unavailable, using hash embeddings: {exc}")
            vector_lists = [self._hash_embed(text) for text in texts]
        self._vectors = vector_lists

        if self.client is None or not chunks:
            return
        try:
            from qdrant_client.models import Distance, VectorParams, PointStruct

            size = len(vector_lists[0]) if vector_lists else EMBEDDING_DIM
            try:
                self.client.recreate_collection(
                    collection,
                    vectors_config=VectorParams(size=size, distance=Distance.COSINE),
                )
            except AttributeError:
                self.client.delete_collection(collection)
                self.client.create_collection(
                    collection,
                    vectors_config=VectorParams(size=size, distance=Distance.COSINE),
                )
            points = [
                PointStruct(
                    id=i,
                    vector=vector_lists[i],
                    payload={**chunk.get("metadata", {}), "text": chunk["text"]},
                )
                for i, chunk in enumerate(chunks)
            ]
            self.client.upsert(collection, points)
        except Exception as exc:
            print(f"  Warning: Qdrant unavailable, using in-memory dense search: {exc}")

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if not self._documents:
            return []
        try:
            query_vector = self._get_encoder().encode(query)
            query_vector = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
        except Exception:
            query_vector = self._hash_embed(query, dim=len(self._vectors[0]) if self._vectors else 384)

        if self.client is not None:
            try:
                response = self.client.query_points(collection, query=query_vector, limit=top_k)
                return [
                    SearchResult(
                        text=pt.payload.get("text", ""),
                        score=float(pt.score),
                        metadata={k: v for k, v in pt.payload.items() if k != "text"},
                        method="dense",
                    )
                    for pt in response.points
                ]
            except Exception:
                pass

        def cosine(a: list[float], b: list[float]) -> float:
            denom = (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0
            return sum(x * y for x, y in zip(a, b)) / denom

        scores = [cosine(query_vector, vector) for vector in self._vectors]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                text=self._documents[i]["text"],
                score=float(scores[i]),
                metadata=self._documents[i].get("metadata", {}),
                method="dense",
            )
            for i in top_indices
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = sum 1/(k + rank + 1)."""
    rrf_scores: dict[str, dict] = {}
    for results in results_list:
        for rank, result in enumerate(results):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    fused = sorted(rrf_scores.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in fused
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF."""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhan vien duoc nghi phep nam")
    print(f"Segmented: {segment_vietnamese('Nhan vien duoc nghi phep nam')}")

