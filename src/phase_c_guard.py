from __future__ import annotations

"""Phase C: PII detection, input/output guardrails, adversarial suite, latency."""

import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE


VN_CCCD_RE = re.compile(r"\b(?:\d{12}|\d{9})\b")
VN_PHONE_RE = re.compile(r"\b0[3-9]\d{8}\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

REFUSAL = (
    "Xin lỗi, tôi không thể thực hiện yêu cầu này. "
    "Tôi chỉ hỗ trợ các câu hỏi an toàn về chính sách nội bộ của công ty."
)


def setup_presidio():
    """Initialize Presidio with Vietnamese CCCD and phone recognizers."""
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits", r"\b\d{9}\b", 0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)
    analyzer = AnalyzerEngine(registry=registry)
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def _regex_pii_scan(text: str) -> dict:
    entities: list[dict] = []
    for entity_type, regex in (
        ("EMAIL_ADDRESS", EMAIL_RE),
        ("VN_CCCD", VN_CCCD_RE),
        ("VN_PHONE", VN_PHONE_RE),
    ):
        for match in regex.finditer(text):
            if entity_type == "VN_CCCD" and VN_PHONE_RE.fullmatch(match.group(0)):
                continue
            entities.append({
                "type": entity_type,
                "text": match.group(0),
                "score": 0.9,
                "start": match.start(),
                "end": match.end(),
            })

    entities.sort(key=lambda item: (item["start"], item["end"]))
    anonymized = text
    for entity in sorted(entities, key=lambda item: item["start"], reverse=True):
        anonymized = (
            anonymized[:entity["start"]]
            + f"<{entity['type']}>"
            + anonymized[entity["end"]:]
        )
    return {"has_pii": bool(entities), "entities": entities, "anonymized": anonymized}


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Scan text for CCCD/CMND, Vietnamese phone numbers, and email addresses."""
    if analyzer is not None and anonymizer is not None:
        try:
            results = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
            entities = [
                {
                    "type": result.entity_type,
                    "text": text[result.start:result.end],
                    "score": round(float(result.score), 3),
                    "start": result.start,
                    "end": result.end,
                }
                for result in results
            ]
            anonymized = anonymizer.anonymize(text=text, analyzer_results=results).text if results else text
            regex_result = _regex_pii_scan(text)
            entity_keys = {(item["type"], item["start"], item["end"]) for item in entities}
            for item in regex_result["entities"]:
                key = (item["type"], item["start"], item["end"])
                if key not in entity_keys:
                    entities.append(item)
            if regex_result["has_pii"] and anonymized == text:
                anonymized = regex_result["anonymized"]
            return {"has_pii": bool(entities), "entities": entities, "anonymized": anonymized}
        except Exception:
            pass

    return _regex_pii_scan(text)


def setup_nemo_rails():
    """Initialize NeMo Guardrails from guardrails/config.yml."""
    from nemoguardrails import LLMRails, RailsConfig

    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    return LLMRails(config)


def _heuristic_block_reason(text: str) -> str | None:
    lowered = text.lower()
    jailbreak_terms = [
        "bỏ qua", "ignore", "forget", "dan", "unrestricted", "system override",
        "admin command", "priority 1", "đóng vai", "không có giới hạn",
        "previous instructions", "system prompt", "system instructions", "training data",
    ]
    pii_request_terms = [
        "cccd của", "số điện thoại của", "email của nhân viên", "thông tin cá nhân",
        "thông tin nhân viên", "tiết lộ lương", "lương tháng của nhân viên",
        "list all employee salaries", "employee records", "confidential hr data",
        "confidential employee data",
    ]
    off_topic_terms = [
        "bài thơ", "mùa thu", "nấu", "phở", "bitcoin", "ethereum", "usd",
        "phương trình", "vi phân", "marvel", "phim", "thời tiết", "cổ phiếu",
        "recipe", "movie",
    ]
    harmful_terms = ["tấn công mạng", "mật khẩu admin", "dump all", "reveal all"]

    if any(term in lowered for term in jailbreak_terms):
        return "jailbreak_or_prompt_injection"
    if any(term in lowered for term in pii_request_terms):
        return "pii_request"
    if any(term in lowered for term in harmful_terms):
        return "harmful_or_confidential_request"
    if any(term in lowered for term in off_topic_terms):
        return "off_topic"
    return None


async def check_input_rail(text: str, rails=None) -> dict:
    """Check whether user input is allowed by input rails."""
    precheck_reason = _heuristic_block_reason(text)
    if precheck_reason:
        return {
            "allowed": False,
            "blocked_reason": precheck_reason,
            "response": REFUSAL,
        }

    if rails is not None:
        try:
            response = await rails.generate_async(messages=[{"role": "user", "content": text}])
            response_text = response if isinstance(response, str) else str(response)
            blocked = any(
                keyword in response_text.lower()
                for keyword in ("xin lỗi", "không thể", "không được phép", "i cannot", "i'm sorry")
            )
            return {
                "allowed": not blocked,
                "blocked_reason": "nemo_input_rail" if blocked else None,
                "response": response_text,
            }
        except Exception as exc:
            return {"allowed": False, "blocked_reason": f"nemo_error: {exc}", "response": REFUSAL}

    return {
        "allowed": True,
        "blocked_reason": None,
        "response": "allowed",
    }


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Check whether an assistant answer is safe to return."""
    pii = pii_scan(answer)
    reason = _heuristic_block_reason(answer)
    if pii["has_pii"] or reason:
        return {
            "safe": False,
            "flagged_reason": "pii_or_sensitive_output" if pii["has_pii"] else reason,
            "final_answer": REFUSAL,
        }

    if rails is not None:
        try:
            response = await rails.generate_async(messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            response_text = response if isinstance(response, str) else str(response)
            flagged = any(
                keyword in response_text.lower()
                for keyword in ("không thể cung cấp", "i cannot", "xin lỗi")
            )
            return {
                "safe": not flagged,
                "flagged_reason": "nemo_output_rail" if flagged else None,
                "final_answer": response_text if flagged else answer,
            }
        except Exception as exc:
            return {"safe": False, "flagged_reason": f"nemo_error: {exc}", "final_answer": REFUSAL}

    return {"safe": True, "flagged_reason": None, "final_answer": answer}


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("run_adversarial_suite must be called from synchronous code")


def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                          analyzer=None, anonymizer=None) -> list[dict]:
    """Run adversarial inputs through PII and input rails."""
    async def _run_all() -> list[dict]:
        results: list[dict] = []
        for item in adversarial_set:
            blocked_by = None
            pii_result = pii_scan(item["input"], analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"

            if blocked_by is None:
                rail_result = await check_input_rail(item["input"], rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            results.append({
                "id": item["id"],
                "category": item["category"],
                "input": item["input"][:120],
                "expected": item["expected"],
                "actual": actual,
                "blocked_by": blocked_by,
                "passed": actual == item["expected"],
            })
        return results

    results = _run_async(_run_all())
    passed = sum(1 for item in results if item["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


def _percentiles(times: list[float]) -> dict:
    if not times:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(times)
    n = len(ordered)

    def pick(frac: float) -> float:
        idx = min(n - 1, max(0, round((n - 1) * frac)))
        return round(ordered[idx], 2)

    return {"p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99)}


def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                        rails=None, analyzer=None, anonymizer=None) -> dict:
    """Measure P50/P95/P99 latency for PII and input rail layers."""
    presidio_times: list[float] = []
    nemo_times: list[float] = []
    total_times: list[float] = []
    inputs = (test_inputs or ["test input"])[:max(1, n_runs)]

    async def _measure() -> None:
        for text in inputs:
            start_total = time.perf_counter()

            start = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - start) * 1000

            total_ms = (time.perf_counter() - start_total) * 1000
            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(total_ms)

    _run_async(_measure())
    total_percentiles = _percentiles(total_times)
    return {
        "presidio_ms": _percentiles(presidio_times),
        "nemo_ms": _percentiles(nemo_times),
        "total_ms": total_percentiles,
        "latency_budget_ok": total_percentiles["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


def save_phase_c_report(path: str = "reports/guard_results.json") -> dict:
    """Run adversarial suite and latency check, then save JSON report."""
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)

    rails = None
    rail_mode = "heuristic_fallback"
    if os.getenv("LAB24_USE_NEMO_RAILS") == "1":
        rails = setup_nemo_rails()
        rail_mode = "nemo_guardrails"

    results = run_adversarial_suite(adversarial_set, rails=rails)
    latency = measure_p95_latency(
        [item["input"] for item in adversarial_set],
        n_runs=len(adversarial_set),
        rails=rails,
    )
    passed = sum(1 for item in results if item["passed"])
    report = {
        "rail_mode": rail_mode,
        "total": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 3) if results else 0.0,
        "results": results,
        "latency": latency,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase C report saved -> {path}")
    return report


if __name__ == "__main__":
    demo = "Nhân viên có CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép."
    print(pii_scan(demo))
    report = save_phase_c_report()
    print(f"Pass rate: {report['passed']}/{report['total']}")
    print(f"Total P95: {report['latency']['total_ms']['p95']}ms")
