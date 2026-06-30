from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Enrich chunks before embedding: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, re, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_MODEL, OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Enriched chunk."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _openai_client():
    if os.getenv("LAB18_OFFLINE") == "1" or not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI

        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception as exc:
        print(f"  Warning: OpenAI client unavailable: {exc}")
        return None


def _json_from_response(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.DOTALL)
    return json.loads(content)


# Technique 1: Chunk Summarization


def summarize_chunk(text: str) -> str:
    """Create a short summary for a chunk."""
    client = _openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Tom tat doan van sau trong 2-3 cau ngan gon bang tieng Viet."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            summary = resp.choices[0].message.content.strip()
            max_len = max(1, len(text) * 2)
            return summary[:max_len].rstrip()
        except Exception as exc:
            print(f"  Warning: OpenAI summarize failed: {exc}")

    sentences = _sentences(text.replace("\n", " "))
    summary = ". ".join(s.rstrip(".") for s in sentences[:2]) + ("." if sentences else "")
    max_len = max(1, len(text) * 2)
    return summary[:max_len].rstrip()


# Technique 2: Hypothesis Question-Answer (HyQA)


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Generate questions this chunk can answer."""
    client = _openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": f"Dua tren doan van, tao {n_questions} cau hoi ma doan van co the tra loi. Moi cau tren 1 dong."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            questions = resp.choices[0].message.content.strip().splitlines()
            return [q.strip().lstrip("0123456789.-) ") for q in questions if q.strip()][:n_questions]
        except Exception as exc:
            print(f"  Warning: OpenAI HyQA failed: {exc}")

    sentences = [s for s in _sentences(text) if len(s) > 10]
    questions = []
    for sentence in sentences[:n_questions]:
        clean = sentence.rstrip(".?!")
        number_match = re.search(r"\d+", clean)
        if number_match:
            questions.append(f"Quy dinh lien quan den {number_match.group(0)} la gi?")
        else:
            questions.append(f"{clean}?")
    return questions


# Technique 3: Contextual Prepend


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Prepend one contextual sentence while preserving the original chunk."""
    client = _openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Viet 1 cau ngan mo ta doan van nay nam o dau trong tai lieu va noi ve chu de gi. Chi tra ve 1 cau."},
                    {"role": "user", "content": f"Tai lieu: {document_title}\n\nDoan van:\n{text}"},
                ],
                max_tokens=80,
            )
            context = resp.choices[0].message.content.strip()
            return f"{context}\n\n{text}" if context else text
        except Exception as exc:
            print(f"  Warning: OpenAI contextual failed: {exc}")

    prefix = f"Trich tu {document_title}." if document_title else "Ngu canh tai lieu noi bo."
    return f"{prefix}\n\n{text}"


# Technique 4: Auto Metadata Extraction


def extract_metadata(text: str) -> dict:
    """Extract lightweight metadata from a chunk."""
    client = _openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": 'Trich xuat metadata tu doan van. Tra ve JSON: {"topic":"...","entities":["..."],"category":"policy|hr|it|finance","language":"vi|en"}'},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return _json_from_response(resp.choices[0].message.content)
        except Exception as exc:
            print(f"  Warning: OpenAI metadata failed: {exc}")

    lowered = text.lower()
    if any(word in lowered for word in ["mat khau", "vpn", "bao mat", "mfa"]):
        category = "it"
    elif any(word in lowered for word in ["luong", "thuong", "chi phi", "tam ung"]):
        category = "finance"
    elif any(word in lowered for word in ["nghi", "nhan vien", "thu viec", "dao tao"]):
        category = "hr"
    else:
        category = "policy"
    title_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    topic = title_match.group(1).strip() if title_match else (_sentences(text)[0][:80] if _sentences(text) else "general")
    entities = sorted(set(re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", text)))[:5]
    return {"topic": topic, "entities": entities, "category": category, "language": "vi"}


# Combined Single-Call Mode


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata."""
    client = _openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": """Phan tich doan van va tra ve JSON:
{
  "summary": "tom tat 2-3 cau",
  "questions": ["cau hoi 1", "cau hoi 2", "cau hoi 3"],
  "context": "1 cau mo ta doan van nam o dau trong tai lieu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""},
                    {"role": "user", "content": f"Tai lieu: {source}\n\nDoan van:\n{text}"},
                ],
                max_tokens=400,
            )
            return _json_from_response(resp.choices[0].message.content)
        except Exception as exc:
            print(f"  Warning: Enrichment API failed: {exc}")

    summary = summarize_chunk(text)
    return {
        "summary": summary,
        "questions": generate_hypothesis_questions(text),
        "context": f"Trich tu {source}. Doan nay noi ve {extract_metadata(text).get('topic', 'chinh sach')}." if source else "Doan nay nam trong tai lieu noi bo.",
        "metadata": extract_metadata(text),
    }


# Full Enrichment Pipeline


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """Run enrichment over a list of {text, metadata} chunks."""
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


if __name__ == "__main__":
    sample = "Nhan vien chinh thuc duoc nghi phep nam 12 ngay lam viec moi nam. So ngay nghi phep tang them 1 ngay cho moi 5 nam tham nien cong tac."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "So tay nhan vien VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")



