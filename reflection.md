# Reflection - Ôn tập kiến thức Lab 24

## 1. Tôi đã học được gì?

Lab này cho thấy một hệ thống RAG production cần hai lớp năng lực: eval để biết hệ thống tốt/xấu ở đâu, và guardrail để bảo vệ hệ thống trước input/output nguy hiểm. Lần chạy mới đã dùng `gpt-4o-mini` thật cho answer generation, RAGAS và judge, nên kết quả phản ánh rõ hơn hành vi của pipeline khi có LLM.

## 2. Tại sao eval quan trọng trong RAG?

RAG có thể sai ở retrieval hoặc generation. Nếu chỉ nhìn câu trả lời cuối, ta khó biết lỗi nằm ở đâu. RAGAS tách lỗi thành các metric như faithfulness, answer relevancy, context precision và context recall. Ở lần chạy LLM thật, context precision cao nhưng faithfulness thấp trên multi-hop/adversarial, nghĩa là retrieval khá ổn nhưng câu trả lời cần bám evidence tốt hơn.

## 3. Hiểu các metric RAGAS

- `faithfulness`: câu trả lời có được hỗ trợ bởi context không. Thấp nghĩa là answer có thể hallucinate hoặc suy diễn quá context.
- `answer_relevancy`: câu trả lời có đúng trọng tâm câu hỏi không.
- `context_precision`: các chunk retrieve có đúng và ít nhiễu không.
- `context_recall`: context có đủ thông tin cần thiết không.

Kết quả factual avg_score 0.8690 tốt, nhưng multi-hop 0.5375 và adversarial 0.5219 cho thấy câu hỏi phức tạp cần retrieval/prompt tốt hơn.

## 4. LLM-as-Judge là gì?

LLM-as-Judge dùng model để so sánh hai câu trả lời. Trong lab, OpenAI judge chọn winner, giải thích reasoning và cho scores. `swap_and_average` đổi thứ tự A/B để phát hiện position bias. Lần chạy này position bias rate là 0.0%, nhưng verbosity bias là 100%, nên vẫn phải cẩn thận với câu trả lời dài.

## 5. Cohen's kappa là gì?

Cohen's kappa đo mức đồng thuận giữa judge và human label sau khi trừ đi đồng thuận do may mắn. Kappa 0.800 là mạnh, nhưng chỉ trên 10 câu nên chưa đủ để kết luận judge luôn đáng tin. Production nên có tập label lớn hơn và kiểm tra định kỳ.

## 6. Guardrail bảo vệ hệ thống như thế nào?

Guardrail chặn vấn đề trước khi vào RAG và kiểm tra output trước khi trả cho user. Phase C dùng PII scan để bắt CCCD, số điện thoại và email. Input rail dùng deterministic pre-check cho attack rõ ràng và NeMo Guardrails cho framework rail. Kết quả adversarial suite là 20/20, P95 latency 3.20ms.

## 7. Nếu deploy production thật thì cần làm gì thêm?

Tôi sẽ thêm citation bắt buộc vào prompt, lưu source chunk kèm answer, thêm metadata version/effective date cho policy cũ mới, và chia multi-hop question thành sub-questions. CI/CD nên fail nếu RAGAS giảm, kappa thấp, adversarial pass rate dưới 90%, hoặc guard P95 latency vượt 500ms.
