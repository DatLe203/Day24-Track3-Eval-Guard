# Failure Cluster Analysis - Phase A

**Student:** Le Huu Dat 
**Date:** 2026-06-30  
**Evaluation mode:** RAGAS executed with OpenAI-backed metrics after fixing `jiter`, `uuid-utils`, and compatible `langchain-core` versions.

---

## 1. Aggregate RAGAS Scores by Distribution

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 0.9000 | 0.2542 | 0.3000 |
| answer_relevancy | 0.7592 | 0.4292 | 0.3541 |
| context_precision | 0.9167 | 0.8333 | 0.8167 |
| context_recall | 0.9000 | 0.6333 | 0.6167 |
| **avg_score** | **0.8690** | **0.5375** | **0.5219** |

---

## 2. Bottom 10 Questions

| Rank | Question ID | Distribution | avg_score | worst_metric | Diagnosis |
|---:|---:|---|---:|---|---|
| 1 | 50 | adversarial | 0.0000 | faithfulness | LLM hallucinating |
| 2 | 35 | multi_hop | 0.1250 | faithfulness | LLM hallucinating |
| 3 | 39 | multi_hop | 0.1250 | faithfulness | LLM hallucinating |
| 4 | 40 | multi_hop | 0.2083 | faithfulness | LLM hallucinating |
| 5 | 25 | multi_hop | 0.2083 | faithfulness | LLM hallucinating |
| 6 | 6 | factual | 0.2500 | faithfulness | LLM hallucinating |
| 7 | 34 | multi_hop | 0.2500 | faithfulness | LLM hallucinating |
| 8 | 48 | adversarial | 0.2500 | faithfulness | LLM hallucinating |
| 9 | 22 | multi_hop | 0.3333 | faithfulness | LLM hallucinating |
| 10 | 21 | multi_hop | 0.3333 | faithfulness | LLM hallucinating |

---

## 3. Failure Cluster Matrix

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 2 | 17 | 7 | 26 |
| answer_relevancy | 14 | 0 | 0 | 14 |
| context_precision | 3 | 1 | 0 | 4 |
| context_recall | 1 | 2 | 3 | 6 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** factual  
**Dominant metric:** faithfulness

With real LLM-generated answers, the main weakness moved from retrieval precision to faithfulness. Context precision is high across all distributions, which means retrieval is often finding relevant chunks. However, multi-hop and adversarial questions still get low faithfulness because the generated answer may not stay tightly grounded in the retrieved context or may miss policy-version nuance.

---

## 5. Suggested Fixes

| Weak metric | Root cause | Suggested fix |
|---|---|---|
| faithfulness | Answer generation not grounded enough | Require citations, quote exact policy snippets, lower temperature |
| answer_relevancy | Some factual answers are too generic | Use direct-answer-first prompt format |
| context_recall | Some multi-hop evidence is incomplete | Retrieve by sub-question and merge evidence |
| context_precision | Mostly healthy after reranking | Keep reranker and add version metadata filters |

---

## 6. Note on Adversarial Distribution

Adversarial avg_score is 0.5219, lower than factual avg_score 0.8690. This is expected because adversarial questions contain version conflicts, negation traps, and policy contradiction traps. Question 50 about personal VPN is the worst case, showing that policy contradiction questions need stronger instruction to refuse outdated or unsafe interpretations.
