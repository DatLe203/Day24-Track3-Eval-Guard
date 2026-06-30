# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Student:** Le Huu Dat
**Date:** 2026-06-30  
**Model:** `gpt-4o-mini` from `LLM_MODEL` / `JUDGE_MODEL` in `.env`  
**Run note:** This report was regenerated with `OPENAI_API_KEY` enabled. Answer generation, chunk enrichment, RAGAS, and pairwise judge used OpenAI-backed execution. Phase C used NeMo Guardrails with deterministic pre-check rules for known high-confidence attacks.

---

## Guard Stack Architecture

```text
User Input
    |
    v
[PII Scan]
    | tool: Presidio-compatible regex recognizers
    | block if: VN_CCCD, VN_PHONE, EMAIL
    v
[Input Guardrail]
    | tool: deterministic attack pre-check + NeMo Guardrails
    | block if: off-topic, jailbreak, prompt injection, PII request
    v
[RAG Pipeline - Day 18]
    | M1 Chunking -> M2 Search -> M3 Rerank -> gpt-4o-mini answer generation
    v
[Output Guardrail]
    | tool: PII/sensitive-output check + optional NeMo output rail
    | action: replace unsafe output with safe refusal
    v
User Response
```

---

## Latency Budget

Measured from `src/phase_c_guard.py` on the 20 adversarial inputs with `LAB24_USE_NEMO_RAILS=1`.

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---:|---:|---:|---:|
| PII Detection | 0.02 | 0.06 | 0.07 | <10ms |
| Input Guardrail | 0.01 | 3.17 | 3.33 | <300ms |
| RAG Pipeline | not measured in guard test | not measured | not measured | <2000ms |
| Output Guardrail | not measured in suite | not measured | not measured | <300ms |
| **Total Guard** | 0.02 | **3.20** | 3.39 | **<500ms** |

**Budget OK:** Yes  
**Comment:** The adversarial suite is dominated by local PII/pre-check blocks, so NeMo latency remains low. Production should separately measure allowed HR-policy queries that pass through NeMo and RAG.

---

## CI/CD Gates

```yaml
- name: Generate Answers
  run: python setup_answers.py
  env:
    LLM_MODEL: gpt-4o-mini
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py

- name: Judge Calibration Gate
  run: python src/phase_b_judge.py
  env:
    LAB24_USE_OPENAI_JUDGE: "1"
    MIN_COHEN_KAPPA: "0.60"

- name: Guardrail Gate
  run: python src/phase_c_guard.py
  env:
    LAB24_USE_NEMO_RAILS: "1"
    MIN_ADV_PASS_RATE: "0.90"
    MAX_GUARD_P95_MS: "500"
```

---

## Actual Lab Results

| Item | Result |
|---|---:|
| RAGAS factual avg_score | 0.8690 |
| RAGAS multi_hop avg_score | 0.5375 |
| RAGAS adversarial avg_score | 0.5219 |
| Worst RAGAS metric | faithfulness |
| Dominant failure distribution | factual |
| Cohen's kappa | 0.800 |
| Judge mode | OpenAI pairwise judge |
| Guard rail mode | NeMo Guardrails + deterministic pre-check |
| Adversarial pass rate | 20/20 |
| Guard P95 latency | 3.20ms |

---

## Production Improvements

1. Add metadata filters for policy version and effective date so adversarial version-conflict questions retrieve the current policy first.
2. Tighten answer generation prompts with citation/evidence requirements because faithfulness is now the dominant weak metric.
3. Measure guard latency on both blocked and allowed traffic; the current adversarial suite mostly exercises blocked paths.
4. Store `ragas_50q.json`, `judge_results.json`, and `guard_results.json` as CI artifacts for regression tracking.
