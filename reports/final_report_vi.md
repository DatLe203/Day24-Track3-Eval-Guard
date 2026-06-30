# Báo cáo cuối Lab 24 - Eval + Guardrail Stack

**Sinh viên:** Le Huu Dat 
**Ngày:** 2026-06-30  
**Model:** `gpt-4o-mini` từ `.env`

## 1. Mục tiêu lab

Lab này xây dựng một stack đánh giá và bảo vệ cho hệ thống RAG. Pipeline gồm Phase A đánh giá RAG bằng RAGAS, Phase B dùng LLM-as-Judge để so sánh câu trả lời, và Phase C dùng guardrails để chặn PII, jailbreak, prompt injection và câu hỏi ngoài phạm vi.

## 2. Tổng quan lần chạy mới

Lần chạy này đã dùng `OPENAI_API_KEY` và `LLM_MODEL=gpt-4o-mini`. `setup_answers.py` chạy lại toàn bộ 50 câu hỏi với LLM thật. Phase A chạy RAGAS thật sau khi sửa dependency `jiter`, `uuid-utils` và đưa `langchain-core` về version tương thích. Phase B chạy OpenAI pairwise judge với `LAB24_USE_OPENAI_JUDGE=1`. Phase C chạy `rail_mode=nemo_guardrails` với NeMo Guardrails và deterministic pre-check cho các attack pattern rõ ràng.

## 3. Phase A - RAGAS Evaluation

| Distribution | Count | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Avg Score |
|---|---:|---:|---:|---:|---:|---:|
| factual | 20 | 0.9000 | 0.7592 | 0.9167 | 0.9000 | 0.8690 |
| multi_hop | 20 | 0.2542 | 0.4292 | 0.8333 | 0.6333 | 0.5375 |
| adversarial | 10 | 0.3000 | 0.3541 | 0.8167 | 0.6167 | 0.5219 |

Kết quả cho thấy factual tốt nhất, còn multi-hop và adversarial khó hơn rõ rệt. Context precision cao nghĩa là retrieval đã lấy được chunk khá liên quan. Điểm yếu chính chuyển sang `faithfulness`: câu trả lời sinh bởi LLM chưa luôn bám chặt vào context, nhất là với câu hỏi cần suy luận nhiều bước hoặc bẫy chính sách.

## 4. Phase B - LLM-as-Judge

| Metric | Result |
|---|---:|
| Judge mode | OpenAI pairwise judge |
| Cohen's kappa | 0.800 |
| Position bias rate | 0.0% |
| Verbosity bias | 100.0% |

Kappa 0.800 cho thấy judge khá khớp với human labels trên bộ 10 câu. Swap-and-average không phát hiện position bias trong 5 cặp mẫu. Tuy nhiên verbosity bias cao: câu thắng thường dài hơn, nên production judge cần prompt phạt thông tin thừa hoặc không có bằng chứng.

## 5. Phase C - Guardrails

| Metric | Result |
|---|---:|
| Rail mode | NeMo Guardrails + deterministic pre-check |
| Adversarial suite passed | 20/20 |
| Pass rate | 100% |
| PII P95 latency | 0.06ms |
| Input rail P95 latency | 3.17ms |
| Total guard P95 latency | 3.20ms |

PII scan chặn CCCD, số điện thoại Việt Nam và email. Input rail chặn yêu cầu PII, jailbreak, prompt injection và off-topic. Pre-check được đặt trước NeMo cho các pattern rất rõ ràng, còn NeMo vẫn là rail framework chính cho stack.

## 6. Nhận xét và cải thiện

Điểm mạnh là hệ thống đã chạy end-to-end với LLM thật, sinh đủ JSON report, analysis report và guard suite đạt 20/20. Điểm cần cải thiện nhất là faithfulness cho multi-hop/adversarial: nên bắt câu trả lời trích dẫn evidence, dùng prompt yêu cầu chỉ trả lời từ context và thêm metadata version policy. Với production, CI nên gate theo RAGAS, Cohen's kappa, adversarial pass rate và guard P95 latency.
