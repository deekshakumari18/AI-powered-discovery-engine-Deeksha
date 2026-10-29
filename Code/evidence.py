"""Structured evidence for prompts and citations."""

from __future__ import annotations

import re
from typing import Any

from hybrid_search import lexical_overlap
from rerank import (
    PROBLEM_DOC,
    PRAISE_DOC,
    PROBLEM_QUERY,
    STATUS_QUERY,
    SUPPORT_QUALITY_QUERY,
    SUPPORT_QUERY,
    TRANSACTION_QUERY,
    active_query_facets,
    portfolio_monitoring_score,
    support_experience_alignment,
    transaction_status_alignment,
)

RECORD_ID = re.compile(r"\b(rec-\d{4})\b", re.IGNORECASE)
ADVICE_MARKERS = re.compile(
    r"\b(you should|i suggest|try to|my advice|recommend that you|"
    r"for beginners,? (?:you|try)|go with)\b",
    re.IGNORECASE,
)
GENERIC_PRAISE = re.compile(
    r"\b(great app|best app|love the app|smooth experience|highly recommend)\b",
    re.IGNORECASE,
)


def classify_voice(text: str) -> str:
    t = text.strip()
    if ADVICE_MARKERS.search(t) and not t.lower().startswith("i "):
        return "advice_or_commenter"
    if re.search(r"\bmy friend\b|\bhe told me\b|\bshe said\b", t, re.I):
        return "about_other_person"
    return "personal_experience"


def to_evidence_item(hit: dict) -> dict[str, Any]:
    text = hit.get("text") or ""
    return {
        "id": hit["id"],
        "source_type": hit.get("source_type") or "source",
        "citation_url": hit.get("citation_url") or "",
        "reviewer": hit.get("reviewer") or hit["id"],
        "text": text,
        "voice": classify_voice(text),
        "relevance": float(hit.get("relevance_score", hit.get("similarity", 0.0))),
        "english_translation": hit.get("english_translation"),
        "section": hit.get("section") or "",
    }


def _topic_signature(hit: dict, lexical_q: str) -> str:
    text = (hit.get("text") or "")[:200].lower()
    if GENERIC_PRAISE.search(text) and not PROBLEM_DOC.search(text):
        return "generic_praise"
    if support_experience_alignment(lexical_q, hit.get("text") or "") >= 0.16:
        return "support"
    if transaction_status_alignment(lexical_q, hit.get("text") or "") >= 0.16:
        return "transaction"
    if PROBLEM_DOC.search(text):
        return "problem"
    return (hit.get("section") or "other")[:32]


def _score_hit(hit: dict, lexical_q: str) -> float:
    rel = float(hit.get("relevance_score", hit.get("similarity", 0.0)))
    lex = lexical_overlap(lexical_q, hit.get("text") or "")
    text = hit.get("text") or ""
    direct = 0.0
    direct += min(0.08, transaction_status_alignment(lexical_q, text))
    direct += min(0.08, support_experience_alignment(lexical_q, text))
    penalty = 0.0
    if GENERIC_PRAISE.search(text) and not PROBLEM_DOC.search(text):
        penalty = 0.06
    return 0.58 * rel + 0.32 * lex + direct - penalty


def select_evidence(
    hits: list[dict],
    lexical_q: str,
    *,
    min_items: int = 3,
    max_items: int = 12,
    floor_ratio: float = 0.38,
    retain_top_retrieved: int = 10,
) -> list[dict]:
    """Select records sent to the LLM (not the full retrieval candidate list).

    Input ``hits`` is the retrieval backend output (deduped parent records, typically
    up to 14), sorted by ``relevance_score`` descending.

    Output is at most ``max_items`` records. Policy, in order of inclusion:

    1. **Retrieval-priority** — top ``retain_top_retrieved`` by retrieval relevance.
    2. **Retrieval-slot** — any record appearing in ``hits[0:14]`` not yet chosen.
    3. **Evidence-score** — remaining slots filled by blended relevance/lexical score
       down to ``floor_ratio`` of the best score (at least ``min_items``).
    4. **Facet backfill** — transaction / portfolio / support aligned records if room.

    Final ordering for the prompt: retrieval-priority first (by retrieval rank), then
    others by evidence score. The cap ``max_items`` must not drop retrieval-priority
    rows in favour of lower-ranked generic matches.
    """
    report = _select_evidence_internal(
        hits,
        lexical_q,
        min_items=min_items,
        max_items=max_items,
        floor_ratio=floor_ratio,
        retain_top_retrieved=retain_top_retrieved,
    )
    return report["prompt_evidence"]


def explain_evidence_selection(
    hits: list[dict],
    lexical_q: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Diagnostics: retrieval candidates vs records actually sent to the LLM."""
    return _select_evidence_internal(hits, lexical_q, **kwargs)


def _select_evidence_internal(
    hits: list[dict],
    lexical_q: str,
    *,
    min_items: int = 3,
    max_items: int = 12,
    floor_ratio: float = 0.38,
    retain_top_retrieved: int = 10,
) -> dict[str, Any]:
    if not hits:
        return {
            "retrieval_candidates": [],
            "prompt_evidence": [],
            "dropped_from_retrieval_top14": [],
            "tiers": {},
        }

    facets = active_query_facets(lexical_q)
    if len(facets) >= 2:
        max_items = max(max_items, 10)
    if (
        SUPPORT_QUERY.search(lexical_q)
        or SUPPORT_QUALITY_QUERY.search(lexical_q)
        or TRANSACTION_QUERY.search(lexical_q)
        or STATUS_QUERY.search(lexical_q)
    ):
        max_items = max(max_items, 10)

    scored: list[tuple[float, dict]] = []
    for hit in hits:
        scored.append((_score_hit(hit, lexical_q), hit))
    scored.sort(key=lambda row: row[0], reverse=True)

    by_retrieval = sorted(
        hits,
        key=lambda h: float(h.get("relevance_score", h.get("similarity", 0.0))),
        reverse=True,
    )
    retrieval_rank = {h["id"]: i + 1 for i, h in enumerate(by_retrieval)}

    chosen: dict[str, dict] = {}
    tier: dict[str, int] = {}

    def remember(hit: dict, score: float, tier_id: int) -> None:
        rid = hit["id"]
        if rid in chosen and tier.get(rid, 99) <= tier_id:
            return
        copy = dict(hit)
        copy["relevance_score"] = score
        copy["_selection_tier"] = tier_id
        chosen[rid] = copy
        tier[rid] = tier_id

    topic_counts: dict[str, int] = {}

    def try_add(hit: dict, score: float, tier_id: int) -> bool:
        rid = hit["id"]
        if rid in chosen:
            return True
        sig = _topic_signature(hit, lexical_q)
        if topic_counts.get(sig, 0) >= 2 and sig == "generic_praise":
            return False
        remember(hit, score, tier_id)
        topic_counts[sig] = topic_counts.get(sig, 0) + 1
        return True

    for hit in by_retrieval[:retain_top_retrieved]:
        try_add(hit, _score_hit(hit, lexical_q), 0)

    for hit in hits[: min(14, len(hits))]:
        try_add(hit, _score_hit(hit, lexical_q), 1)

    top = scored[0][0] if scored else 0.0
    threshold = top * floor_ratio if top > 0 else 0.0

    for score, hit in scored:
        if len(chosen) >= max_items + 8:
            break
        if hit["id"] in chosen:
            continue
        if score >= threshold or len(chosen) < min_items:
            try_add(hit, score, 2)

    for score, hit in scored:
        if len(chosen) >= max_items + 8:
            break
        if hit["id"] in chosen:
            continue
        text = hit.get("text") or ""
        if transaction_status_alignment(lexical_q, text) >= 0.18:
            try_add(hit, score, 3)
        if portfolio_monitoring_score(lexical_q, text) >= 0.18:
            try_add(hit, score, 3)
        if support_experience_alignment(lexical_q, text) >= 0.18:
            try_add(hit, score, 3)

    priority_ids: list[str] = []
    for hit in by_retrieval[:retain_top_retrieved]:
        if hit["id"] in chosen:
            priority_ids.append(hit["id"])
    for hit in hits[: min(14, len(hits))]:
        if hit["id"] in chosen and hit["id"] not in priority_ids:
            priority_ids.append(hit["id"])

    ordered: list[dict] = []
    for rid in priority_ids:
        if len(ordered) >= max_items:
            break
        ordered.append(chosen[rid])

    rest = [chosen[rid] for rid in chosen if rid not in {h["id"] for h in ordered}]
    rest.sort(
        key=lambda h: (
            tier.get(h["id"], 9),
            -float(h.get("relevance_score", 0.0)),
        ),
    )
    for hit in rest:
        if len(ordered) >= max_items:
            break
        ordered.append(hit)

    prompt_ids = [h["id"] for h in ordered]
    top14_ids = [h["id"] for h in hits[: min(14, len(hits))]]
    dropped = [rid for rid in top14_ids if rid not in prompt_ids]

    return {
        "retrieval_candidates": [h["id"] for h in hits],
        "retrieval_top14": top14_ids,
        "prompt_evidence": ordered,
        "prompt_ids": prompt_ids,
        "dropped_from_retrieval_top14": dropped,
        "tiers": {rid: tier.get(rid) for rid in prompt_ids},
        "max_items": max_items,
    }


def prompt_block(items: list[dict[str, Any]]) -> str:
    blocks = []
    for item in items:
        extra = ""
        if item.get("english_translation"):
            extra = f"\nEnglish: {item['english_translation']}"
        blocks.append(
            f"[{item['id']}] voice={item['voice']} type={item['source_type']}\n"
            f"{item['text']}{extra}"
        )
    return "\n\n".join(blocks)


def parse_cited_record_ids(body: str, valid_ids: set[str]) -> list[str]:
    seen: list[str] = []
    for match in RECORD_ID.finditer(body):
        rid = match.group(1).lower()
        m = re.match(r"^rec-(\d+)$", rid)
        if not m:
            continue
        rid = f"rec-{int(m.group(1)):04d}"
        if rid in valid_ids and rid not in seen:
            seen.append(rid)
    return seen


def format_citations_for_ids(
    items: list[dict[str, Any]],
    cited_ids: list[str],
) -> str:
    by_id = {item["id"]: item for item in items}
    lines: list[str] = []
    targets = cited_ids if cited_ids else [item["id"] for item in items[:3]]
    for rid in targets:
        item = by_id.get(rid)
        if not item:
            continue
        title = item.get("reviewer") or rid
        url = item.get("citation_url") or ""
        voice = item.get("voice") or "personal_experience"
        if url:
            lines.append(f"**{rid}** · {item.get('source_type')} · {voice} · {title}\n{url}")
        else:
            excerpt = " ".join((item.get("text") or "").split())
            if len(excerpt) > 320:
                excerpt = excerpt[:320].rsplit(" ", 1)[0] + "…"
            lines.append(
                f"**{rid}** · {item.get('source_type')} · {voice} · {title}\n{excerpt}"
            )
    return "\n\n".join(lines) if lines else ""


def fallback_cited_ids(items: list[dict[str, Any]], lexical_q: str, limit: int = 3) -> list[str]:
    ranked = sorted(
        items,
        key=lambda item: (
            item.get("relevance", 0.0),
            lexical_overlap(lexical_q, item.get("text") or ""),
        ),
        reverse=True,
    )
    out: list[str] = []
    for item in ranked:
        text = item.get("text") or ""
        if PROBLEM_QUERY.search(lexical_q) and PRAISE_DOC.search(text) and not PROBLEM_DOC.search(text):
            continue
        out.append(item["id"])
        if len(out) >= limit:
            break
    if not out:
        out = [item["id"] for item in ranked[:limit]]
    return out
