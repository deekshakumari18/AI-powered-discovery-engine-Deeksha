"""Stage-by-stage trace for one record vs a query (diagnostics only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedding import get_model
from hybrid_search import expand_query, hybrid_score, lexical_overlap
from retrieval import (
    MIN_SIMILARITY,
    _as_review,
    _best_by_review,
    _filter_sip_mf_reviews,
    _hits_from_result,
    _merge_lexical_supplements,
    _parent_id,
    _relevance_boost,
    _sip_mf_relevant,
    _units_by_id,
)
from vector_store import COLLECTION_NAME, get_client


def trace_record(query: str, record_id: str, lexical_question: str | None = None) -> dict:
    import numpy as np

    lexical_q = expand_query(lexical_question or query)
    expanded = expand_query(query)
    units = _units_by_id()
    unit = units.get(record_id)
    if not unit:
        return {"error": "record not in corpus", "record_id": record_id}

    collection = get_client().get_collection(COLLECTION_NAME)
    vector = get_model().encode([expanded], normalize_embeddings=True)[0]
    query_v = np.asarray(vector, dtype=np.float32)

    total = collection.count()
    result = collection.query(
        query_embeddings=[vector.tolist()],
        n_results=total,
        include=["documents", "metadatas", "distances"],
    )
    hits = _hits_from_result(result)
    vector_rank = next(
        (
            i + 1
            for i, h in enumerate(hits)
            if _parent_id(h["id"]) == record_id
        ),
        None,
    )

    hits = _merge_lexical_supplements(hits, lexical_q, units, collection, query_v)
    in_after_supp = any(_parent_id(h["id"]) == record_id for h in hits)

    doc = unit.get("embed_text") or unit.get("text") or ""
    lex_doc = lexical_overlap(lexical_q, doc)

    for hit in hits:
        if _parent_id(hit["id"]) != record_id:
            continue
        parent = record_id
        u = units.get(parent, {})
        d = u.get("embed_text") or u.get("text") or hit["text"]
        lex = max(lexical_overlap(lexical_q, d), lexical_overlap(lexical_q, hit["text"]))
        vs = hit["similarity"]
        hit["vector_similarity"] = vs
        hit["similarity"] = hybrid_score(vs, lex)

    by_review = _best_by_review(hits)
    hit_rec = next((h for h in by_review if _parent_id(h["id"]) == record_id), None)
    hybrid_rank = next(
        (i + 1 for i, h in enumerate(by_review) if _parent_id(h["id"]) == record_id),
        None,
    )

    filtered_reason = None
    final_rank = None
    stage = "not_in_pool"
    if hit_rec:
        review = _as_review(hit_rec, units)
        if hit_rec["similarity"] < MIN_SIMILARITY:
            filtered_reason = "below MIN_SIMILARITY"
        elif not _sip_mf_relevant(review["text"]):
            pq = lexical_question or query
            kept = _filter_sip_mf_reviews([review], pq)
            if not kept:
                filtered_reason = "sip_mf_filter"
        else:
            overlap = max(
                lexical_overlap(lexical_q, review["text"]),
                lexical_overlap(lexical_q, unit.get("embed_text") or ""),
            )
            vs = hit_rec.get("vector_similarity", hit_rec["similarity"])
            if overlap < 0.06 and vs < 0.42:
                filtered_reason = "lexical_gate"

    if hit_rec and not filtered_reason:
        reviews = []
        for h in by_review:
            if h["similarity"] < MIN_SIMILARITY:
                continue
            r = _as_review(h, units)
            u2 = units.get(r["id"], {})
            ov = max(
                lexical_overlap(lexical_q, r["text"]),
                lexical_overlap(lexical_q, u2.get("embed_text") or ""),
            )
            v2 = h.get("vector_similarity", h["similarity"])
            if ov < 0.06 and v2 < 0.42:
                continue
            r["similarity"] = _relevance_boost(
                lexical_question or query, lexical_q, u2, r, r["similarity"]
            )
            reviews.append(r)
        reviews = _filter_sip_mf_reviews(reviews, lexical_question or query)
        reviews.sort(key=lambda x: x["similarity"], reverse=True)
        for i, r in enumerate(reviews):
            if r["id"] == record_id:
                final_rank = i + 1
                stage = "final_top14" if final_rank <= 14 else "final_below_top14"
                break
        if final_rank is None:
            filtered_reason = filtered_reason or "dropped_in_filter_pass"

    from retrieval import retrieve as full_retrieve

    full_hits = full_retrieve(query, lexical_question=lexical_question or query)
    full_rank = next(
        (i + 1 for i, h in enumerate(full_hits) if h["id"] == record_id),
        None,
    )

    return {
        "record_id": record_id,
        "query": query[:120],
        "lexical_overlap_full_doc": round(lex_doc, 4),
        "vector_chunk_rank": vector_rank,
        "in_pool_after_lexical_supplement": in_after_supp,
        "hybrid_rank_among_records": hybrid_rank,
        "hit_hybrid_score": round(hit_rec["similarity"], 4) if hit_rec else None,
        "sip_mf_relevant": _sip_mf_relevant(_as_review(hit_rec, units)["text"])
        if hit_rec
        else None,
        "final_rank_legacy_vector_only": final_rank,
        "final_rank_after_rerank": full_rank,
        "stage": "final_top14"
        if full_rank and full_rank <= 14
        else ("final_below_top14" if full_rank else "not_in_pool"),
        "filtered_reason": filtered_reason,
    }


if __name__ == "__main__":
    cases = [
        (
            "What problems do beginners describe when starting on the app, compared with complaints about wrong balances on the dashboard?",
            "rec-0158",
        ),
        (
            "What worries users about tracking their portfolio after they start a SIP?",
            "rec-0192",
        ),
        (
            "What do people report about failed KYC or verification steps when trying to invest in mutual funds?",
            "rec-0032",
        ),
    ]
    for q, rid in cases:
        print(json.dumps(trace_record(q, rid, lexical_question=q), indent=2))
