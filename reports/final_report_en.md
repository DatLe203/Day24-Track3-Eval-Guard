# Final Report - Lab 24 Eval + Guardrail Stack

**Student:** Le Huu Dat  
**Date:** 2026-06-30  
**Model:** `gpt-4o-mini` from `.env`

## 1. Objective

This lab builds a production-style evaluation and guardrail stack for a RAG system. Phase A evaluates answer quality with RAGAS, Phase B calibrates an LLM-as-Judge workflow, and Phase C protects the system with PII detection, input guardrails, output checks, adversarial testing, and latency measurement.

## 2. New Run Overview

This run used `OPENAI_API_KEY` and `LLM_MODEL=gpt-4o-mini`. `setup_answers.py` regenerated all 50 answers with OpenAI-backed answer generation. Phase A ran real RAGAS after fixing `jiter`, `uuid-utils`, and compatible `langchain-core` versions. Phase B used OpenAI pairwise judge with `LAB24_USE_OPENAI_JUDGE=1`. Phase C used `rail_mode=nemo_guardrails`, combining NeMo Guardrails with deterministic pre-checks for high-confidence attack patterns.

## 3. Phase A - RAGAS Evaluation

| Distribution | Count | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Avg Score |
|---|---:|---:|---:|---:|---:|---:|
| factual | 20 | 0.9000 | 0.7592 | 0.9167 | 0.9000 | 0.8690 |
| multi_hop | 20 | 0.2542 | 0.4292 | 0.8333 | 0.6333 | 0.5375 |
| adversarial | 10 | 0.3000 | 0.3541 | 0.8167 | 0.6167 | 0.5219 |

Factual questions perform best. Multi-hop and adversarial questions remain significantly harder. Context precision is high, so retrieval is often finding relevant evidence. The dominant weakness is faithfulness: generated answers are not always tightly grounded in the retrieved context, especially on reasoning-heavy or adversarial policy questions.

## 4. Phase B - LLM-as-Judge

| Metric | Result |
|---|---:|
| Judge mode | OpenAI pairwise judge |
| Cohen's kappa | 0.800 |
| Position bias rate | 0.0% |
| Verbosity bias | 100.0% |

A kappa of 0.800 indicates strong agreement with human labels on the small 10-question calibration set. Swap-and-average found no position inconsistency in the five pairwise samples. The high verbosity bias means the judge tends to prefer longer answers, so production prompts should penalize unsupported extra detail.

## 5. Phase C - Guardrails

| Metric | Result |
|---|---:|
| Rail mode | NeMo Guardrails + deterministic pre-check |
| Adversarial suite passed | 20/20 |
| Pass rate | 100% |
| PII P95 latency | 0.06ms |
| Input rail P95 latency | 3.17ms |
| Total guard P95 latency | 3.20ms |

The guard stack detects Vietnamese CCCD/CMND numbers, Vietnamese phone numbers, and email addresses. It blocks PII requests, jailbreak attempts, prompt injection, and off-topic inputs. The deterministic pre-check handles obvious attacks before NeMo, while NeMo remains the configured guardrail framework.

## 6. Production Readiness

The stack now runs end to end with OpenAI-backed generation, RAGAS evaluation, OpenAI judge, and NeMo guardrails. The main improvement area is faithfulness for multi-hop and adversarial questions. Recommended next steps are citation-enforced prompting, version-aware policy metadata, sub-question retrieval for multi-hop questions, and CI gates for RAGAS, Cohen's kappa, adversarial pass rate, and guard P95 latency.
