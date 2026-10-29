"""Lightweight claim-to-source checks (safeguard, not semantic proof)."""

from __future__ import annotations

import re

from hybrid_search import lexical_overlap, token_set
from evidence import RECORD_ID

MIN_SUPPORT_OVERLAP = 0.07
MIN_ANCHOR_TOKENS = 2


def _normalize_rid(raw: str) -> str | None:
    m = re.match(r"^rec-(\d+)$", raw.lower())
    if not m:
        return None
    return f"rec-{int(m.group(1)):04d}"


def _sentence_for_cite(body: str, cite_start: int) -> str:
    chunk = body[max(0, cite_start - 320) : cite_start + 80]
    parts = re.split(r"(?<=[.!?])\s+", chunk)
    for part in reversed(parts):
        if "[rec-" in part.lower():
            return part.strip()
    return chunk.strip()


def _content_words(text: str) -> set[str]:
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "that",
        "this",
        "with",
        "from",
        "they",
        "their",
        "them",
        "have",
        "has",
        "was",
        "were",
        "been",
        "also",
        "about",
        "when",
        "what",
        "review",
        "reviews",
        "groww",
        "user",
        "users",
        "people",
        "report",
        "reported",
        "say",
        "said",
    }
    return {t for t in token_set(text) if t not in stop and len(t) >= 4}


def claim_supported_by_source(sentence: str, source_text: str) -> tuple[bool, float]:
    cleaned = RECORD_ID.sub("", sentence).strip()
    if len(cleaned) < 12:
        return True, 1.0
    overlap = lexical_overlap(cleaned, source_text)
    claim_words = _content_words(cleaned)
    source_words = _content_words(source_text)
    anchors = claim_words & source_words
    if overlap >= MIN_SUPPORT_OVERLAP and len(anchors) >= MIN_ANCHOR_TOKENS:
        return True, overlap
    if len(anchors) >= 3:
        return True, overlap
    return False, overlap


def validate_answer_citations(
    body: str,
    items: list[dict],
    *,
    valid_ids: set[str] | None = None,
) -> list[dict]:
    """Return issues for citations whose adjacent sentence poorly matches the source."""
    by_text = {item["id"]: item.get("text") or "" for item in items}
    allowed = valid_ids or set(by_text)
    issues: list[dict] = []
    for match in RECORD_ID.finditer(body):
        rid = _normalize_rid(match.group(1))
        if not rid or rid not in allowed:
            continue
        source = by_text.get(rid, "")
        if not source:
            issues.append({"id": rid, "reason": "missing_source_text"})
            continue
        sentence = _sentence_for_cite(body, match.start())
        ok, overlap = claim_supported_by_source(sentence, source)
        if not ok:
            issues.append(
                {
                    "id": rid,
                    "reason": "weak_lexical_support",
                    "overlap": round(overlap, 4),
                    "sentence": sentence[:200],
                }
            )
    return issues


def filter_cited_ids_by_validation(
    body: str,
    items: list[dict],
    cited_ids: list[str],
) -> tuple[list[str], list[dict]]:
    issues = validate_answer_citations(body, items)
    weak = {row["id"] for row in issues if row["reason"] == "weak_lexical_support"}
    filtered = [rid for rid in cited_ids if rid not in weak]
    if not filtered:
        filtered = cited_ids[:]
    return filtered, issues
