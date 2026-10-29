"""Stage trace for known-gap paraphrases (evaluation only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bm25_index import get_bm25_index
from evidence import select_evidence, prompt_block, to_evidence_item
from hybrid_search import expand_query, lexical_focus_query
from retrieval import (
    MIN_SIMILARITY,
    retrieve,
    _units_by_id,
)
from rerank import (
    combined_rerank_score,
    facet_alignment_score,
    support_experience_alignment,
    transaction_status_alignment,
)


def trace_record(question: str, record_id: str) -> dict:
    lexical_q = expand_query(question)
    units = _units_by_id()
    unit = units.get(record_id)
    if not unit:
        return {"record_id": record_id, "error": "not_in_corpus"}

    hits = retrieve(question, lexical_question=question)
    hit_ids = [h["id"] for h in hits]
    retrieval_rank = hit_ids.index(record_id) + 1 if record_id in hit_ids else None

    import retrieval as rmod

    deep_rank = None
    prev = rmod.MAX_RECORDS
    rmod.MAX_RECORDS = 200
    try:
        deep = retrieve(question, lexical_question=question)
        deep_ids = [h["id"] for h in deep]
        deep_rank = deep_ids.index(record_id) + 1 if record_id in deep_ids else None
    finally:
        rmod.MAX_RECORDS = prev

    doc = unit.get("embed_text") or unit.get("text") or ""
    bm25 = get_bm25_index()
    bm_a = bm25.normalized_score_for(lexical_q, record_id)
    bm_b = bm25.normalized_score_for(lexical_focus_query(lexical_q), record_id)
    facet = facet_alignment_score(question, doc)
    tsa = transaction_status_alignment(question, doc)
    sea = support_experience_alignment(question, doc)

    in_hit = next((h for h in hits if h["id"] == record_id), None)
    hybrid = in_hit.get("similarity") if in_hit else None
    rel = in_hit.get("relevance_score") if in_hit else None
    if in_hit:
        from_pool = in_hit.get("from_vector_pool", True)
        rel_r = combined_rerank_score(
            float(in_hit.get("similarity", 0)),
            max(bm_a, bm_b),
            facet,
            question,
            doc,
            from_vector_pool=from_pool,
        )
    else:
        rel_r = None

    chosen = select_evidence(hits, lexical_q)
    prompt_ids = [h["id"] for h in chosen]
    in_prompt = record_id in prompt_ids
    prompt_rank = prompt_ids.index(record_id) + 1 if in_prompt else None

    stage = "ok"
    if deep_rank is None:
        stage = "never_in_kept_pool"
    elif retrieval_rank is None and deep_rank is not None:
        stage = "rerank_below_top14"
    elif retrieval_rank is None:
        stage = "filtered_before_final_14"
    elif not in_prompt:
        stage = "retrieved_but_dropped_by_evidence_selection"

    return {
        "record_id": record_id,
        "question": question[:100],
        "retrieval_rank_top14": retrieval_rank,
        "retrieval_rank_deep": deep_rank,
        "in_final_retrieve_count": len(hits),
        "hybrid_similarity": hybrid,
        "relevance_score": rel,
        "rerank_recomputed": rel_r,
        "bm25_max": round(max(bm_a, bm_b), 4),
        "facet": facet,
        "transaction_status_alignment": round(tsa, 3),
        "support_experience_alignment": round(sea, 3),
        "in_llm_prompt": in_prompt,
        "prompt_rank": prompt_rank,
        "prompt_evidence_count": len(chosen),
        "prompt_evidence_ids": prompt_ids,
        "failure_stage": stage,
    }


def trace_question(question: str, label_ids: list[str]) -> dict:
    hits = retrieve(question, lexical_question=question)
    lexical_q = expand_query(question)
    chosen = select_evidence(hits, lexical_q)
    return {
        "question": question,
        "retrieve_top14": [h["id"] for h in hits[:14]],
        "prompt_evidence_ids": [h["id"] for h in chosen],
        "prompt_record_count": len(chosen),
        "labels": {rid: trace_record(question, rid) for rid in label_ids},
    }


KNOWN = [
    (
        "payment_status_paraphrase",
        "What happens when instalments are debited but payment or SIP status stays pending?",
        ["rec-0285", "rec-0310", "rec-0321", "rec-0306"],
    ),
    (
        "support_mixed_experiences",
        "Do reviews describe Groww support as responsive or hard to reach?",
        ["rec-0032", "rec-0047", "rec-0325", "rec-0310"],
    ),
]


if __name__ == "__main__":
    out = []
    for name, q, labels in KNOWN:
        out.append({"case": name, **trace_question(q, labels)})
    print(json.dumps(out, indent=2))
