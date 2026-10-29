"""Deep retrieval trace for tests and blind eval (not exposed in UI)."""

from __future__ import annotations

from typing import Any

from hybrid_search import expand_query, lexical_overlap, question_supported_by_texts
from intent import is_corpus_research, should_block_advice, should_block_returns


def trace_retrieval(
    question: str,
    *,
    history: list[dict] | None = None,
    lexical_question: str | None = None,
) -> dict[str, Any]:
    """Run guard + retrieve with stage notes. Imports retrieve lazily to avoid cycles."""
    from hybrid_search import contextualize_question
    from retrieval import (
        MIN_SIMILARITY,
        _filter_sip_mf_reviews,
        _sip_mf_relevant,
        guard,
        retrieve,
    )

    lexical_q = lexical_question or question
    search_q = contextualize_question(question, history) if history else question
    report: dict[str, Any] = {
        "question": question,
        "search_question": search_q,
        "expanded_query": expand_query(search_q),
        "lexical_query": expand_query(lexical_q),
        "corpus_research": is_corpus_research(lexical_q, has_history=bool(history)),
        "would_block_returns": should_block_returns(lexical_q),
        "would_block_advice": should_block_advice(lexical_q),
        "guard_message": guard(lexical_q),
    }
    if report["guard_message"]:
        report["final_hits"] = []
        report["stage"] = "guard"
        return report

    hits = retrieve(search_q, lexical_question=lexical_q)
    report["stage"] = "ok" if hits else "empty"
    report["hit_count"] = len(hits)
    report["top_similarity"] = hits[0]["similarity"] if hits else 0.0
    report["final_hits"] = [
        {
            "id": h["id"],
            "source_type": h.get("source_type"),
            "similarity": round(h["similarity"], 4),
            "citation_url": (h.get("citation_url") or "")[:120],
            "sip_mf_relevant": _sip_mf_relevant(h.get("text") or ""),
        }
        for h in hits[:14]
    ]
    if hits:
        report["anchor_ok"] = question_supported_by_texts(
            lexical_q, [h["text"] for h in hits]
        )
    else:
        report["anchor_ok"] = None
    report["min_similarity"] = MIN_SIMILARITY
    return report
