# LLM Judge Bias Report - Phase B

**Student:** Le Huu Dat  
**Date:** 2026-06-30  
**Judge model:** gpt-4o-mini  
**Run mode:** OpenAI pairwise judge enabled with `LAB24_USE_OPENAI_JUDGE=1`.

---

## 1. Pairwise Judge Results

The judge compares Answer A and Answer B using accuracy, completeness, and conciseness. This run used OpenAI for the pairwise comparisons and kept swap-and-average to reduce position bias.

| # | Question summary | Winner | Reasoning summary |
|---:|---|---|---|
| 1 | Marriage leave days | A | A gives the specific leave entitlement |
| 2 | 55M equipment approval | A | A correctly requires CEO approval |
| 3 | Tet bonus minimum | A | A gives the concrete one-month salary rule |
| 4 | Senior tenure leave and salary | A | A includes both leave days and salary range |
| 5 | Training reimbursement | B | B more clearly states 100% repayment before 12 months |

---

## 2. Swap-and-Average Results

| # | Pass 1 Winner | Pass 2 Winner converted back | Final | Position Consistent? |
|---:|---|---|---|---|
| 1 | A | A | A | Yes |
| 2 | A | A | A | Yes |
| 3 | A | A | A | Yes |
| 4 | A | A | A | Yes |
| 5 | B | B | B | Yes |

**Position bias rate:** 0.0%  
**Interpretation:** No position inconsistency appeared in this five-pair sample. Swap-and-average is still kept because LLM judges can prefer the first answer on harder or more ambiguous pairs.

---

## 3. Cohen's Kappa Analysis

| Question ID | Human Label | Judge Label | Agree? |
|---:|---:|---:|---|
| 1 | 1 | 1 | Yes |
| 5 | 0 | 0 | Yes |
| 12 | 1 | 1 | Yes |
| 21 | 1 | 1 | Yes |
| 23 | 1 | 1 | Yes |
| 29 | 0 | 0 | Yes |
| 33 | 1 | 0 | No |
| 41 | 0 | 0 | Yes |
| 46 | 1 | 1 | Yes |
| 50 | 0 | 0 | Yes |

**Cohen's kappa:** 0.800  
**Interpretation:** Strong agreement on this small calibration set, with one disagreement on Question 33. A larger labeled set is needed before treating the judge as a production quality gate.

---

## 4. Verbosity Bias

| Metric | Value |
|---|---:|
| A wins and A is longer | 4 |
| B wins and B is longer | 1 |
| Total decisive cases | 5 |
| Verbosity bias rate | 100.0% |

The winning answer is longer in every decisive sample. This can be valid when the longer answer contains the needed facts, but production judge prompts should explicitly penalize unsupported extra detail.

---

## 5. General Reflection

OpenAI judge gives useful rationales and stable pairwise results in this run. Cohen's kappa indicates good alignment with human labels, while verbosity bias reminds us that judge quality still needs calibration. The judge should be used as an automated signal, not as absolute truth.
