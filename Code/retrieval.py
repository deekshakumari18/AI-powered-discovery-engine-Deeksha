"""Phase 5 — Retrieval backend.

Guards the question, embeds it with all-MiniLM-L6-v2, queries ChromaDB,
then asks Groq for a short cited answer.

Does not store the question. UI is out of scope.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from embedding import MODEL_ID, get_model
from hybrid_search import (
    contextualize_question,
    expand_query,
    lexical_focus_query,
    hybrid_score,
    lexical_overlap,
    question_supported_by_texts,
    token_set,
)
from bm25_index import get_bm25_index
from citation_validate import filter_cited_ids_by_validation
from evidence import (
    fallback_cited_ids,
    format_citations_for_ids,
    parse_cited_record_ids,
    prompt_block,
    select_evidence,
    to_evidence_item,
)
from rerank import (
    PROBLEM_QUERY,
    MONITORING_SIGNAL,
    SIP_STOCK_QUERY,
    STATUS_QUERY,
    TRACKING_QUERY,
    TRANSACTION_QUERY,
    combined_rerank_score,
    document_specific_boost,
    facet_alignment_score,
    facet_overlap_required,
    motivation_expectation_alignment,
    portfolio_monitoring_score,
    sip_stock_journey_alignment,
    support_experience_alignment,
    SUPPORT_QUALITY_QUERY,
    SUPPORT_QUERY,
    transaction_status_alignment,
)
from vector_store import COLLECTION_NAME, get_client

TOP_K = 4
MIN_SIMILARITY = 0.24
GLOBAL_CHUNK_CANDIDATES = 150
LEXICAL_SUPPLEMENT_UNITS = 36
MIN_LEXICAL_SUPPLEMENT = 0.11
MAX_RECORDS = 14
LOADED_PATH = PROJECT_ROOT / "Data" / "raw" / "loaded.json"
last_hits: list[dict] = []
EDUCATION_URL = "https://investor.sebi.gov.in/"
FACTSHEET_URL = "https://www.amfiindia.com/research-information/other-data/factsheet"

GREETING = re.compile(
    r"^(?:hi|hello|hey|hiya|yo|namaste|good\s+(?:morning|afternoon|evening)|"
    r"how\s+are\s+you|thanks|thank\s+you)(?:\s+there)?[!.?\s]*$",
    re.IGNORECASE,
)
EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE)
AADHAAR = re.compile(r"(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)")
PHONE = re.compile(
    r"(?:\+91[\s\-]*[6-9]\d{9}\b"
    r"|(?:phone|mobile|mob|whatsapp)\s*[:\-]?\s*[6-9]\d{9}\b"
    r"|[6-9]\d{4}[\s\-]\d{5}\b)",
    re.IGNORECASE,
)
OTP = re.compile(
    r"\b(?:otp|one[-\s]?time(?:\s+password)?)\s*[:\-]?\s*\d{4,8}\b",
    re.IGNORECASE,
)
ACCOUNT = re.compile(
    r"\b(?:account|a/?c|acct)\s*(?:no\.?|number|#)\s*[:\-]?\s*\d{8,18}\b"
    r"|\b[A-Z]{4}0[A-Z0-9]{6}\b",
    re.IGNORECASE,
)
from intent import should_block_advice, should_block_returns

MF_TOPIC = re.compile(
    r"\b(?:sip|sips|systematic investment|mutual funds?|\bmf\b|folio|instal+ments?|"
    r"autopay|redeem(?:ed|ption)?|lumpsum|\bnav\b|direct plan|index fund)\b",
    re.IGNORECASE,
)
STOCK_TOPIC = re.compile(
    r"\b(?:stock trading|trading account|share prices?|equity trading|intraday|"
    r"day trading|\bf&o\b|\bfno\b|options?\b|futures\b|market opens|when the market opens|"
    r"\bnifty\b|\bsensex\b|demat\b|brokerage|candlestick|profit/loss|incorrect share|"
    r"share trading)\b",
    re.IGNORECASE,
)
MF_EXPERIENCE = re.compile(
    r"\b(?:start(?:ed|ing)?|continu\w+|paus\w+|stop\w*|cancel\w*|redeem\w*|withdraw\w*|"
    r"invest(?:ed|ing|ment)?|instal+ments?|autopay|modify(?:ing)?|track\w*|debit\w*)\b"
    r".{0,50}\b(?:sip|sips|mutual funds?|\bmf\b)\b"
    r"|\b(?:sip|sips|mutual funds?|\bmf\b)\b.{0,50}\b(?:start(?:ed|ing)?|continu\w+|"
    r"paus\w+|stop\w*|cancel\w*|redeem\w*|withdraw\w*|invest(?:ed|ing|ment)?|"
    r"instal+ments?|autopay|modify(?:ing)?|debit\w*)\b",
    re.IGNORECASE | re.DOTALL,
)


def _has_pii(question: str) -> bool:
    return any(
        pattern.search(question)
        for pattern in (EMAIL, PAN, AADHAAR, PHONE, OTP, ACCOUNT)
    )


def greeting_reply(question: str) -> str | None:
    text = question.strip()
    if not GREETING.match(text):
        return None
    lower = text.lower().strip("!.? ")
    if lower.startswith("good morning"):
        return "Good morning."
    if lower.startswith("good afternoon"):
        return "Good afternoon."
    if lower.startswith("good evening"):
        return "Good evening."
    if lower.startswith("thank"):
        return "You're welcome."
    if lower.startswith("how are"):
        return "I'm doing well."
    if lower.startswith("namaste"):
        return "Namaste."
    return "Hello."


def guard(question: str) -> str | None:
    text = question.strip()
    if not text:
        return "Ask a question about the documents."
    if _has_pii(text):
        return (
            "We do not accept personal data such as PAN, Aadhaar, account numbers, "
            "OTPs, emails, or phone numbers. Ask a general question about the documents."
        )
    if should_block_returns(text):
        return (
            "This tool does not compute or compare returns. "
            f"See the official factsheet: {FACTSHEET_URL}"
        )
    if should_block_advice(text):
        return (
            "This tool shares facts from the documents only and does not give "
            f"investment or portfolio advice. For investor education, see {EDUCATION_URL}"
        )
    return None


def _cosine_from_l2(distance: float) -> float:
    score = 1.0 - (distance * distance) / 2.0
    return max(0.0, min(1.0, score))


def _hits_from_result(result: dict) -> list[dict]:
    hits = []
    for chunk_id, text, meta, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        meta = meta or {}
        hits.append(
            {
                "id": chunk_id,
                "text": text,
                "source_name": meta.get("source_name") or "",
                "section": meta.get("section") or "",
                "citation_url": meta.get("citation_url") or "",
                "last_updated": meta.get("last_updated") or "",
                "source_type": meta.get("source_type") or "",
                "parent_id": meta.get("parent_id") or "",
                "similarity": _cosine_from_l2(float(distance)),
            }
        )
    return hits


def _parent_id(chunk_id: str) -> str:
    match = re.match(r"^(rec-\d+)-c\d+$", chunk_id)
    return match.group(1) if match else chunk_id


def _units_by_id() -> dict[str, dict]:
    if not LOADED_PATH.is_file():
        return {}
    payload = json.loads(LOADED_PATH.read_text(encoding="utf-8"))
    return {unit["id"]: unit for unit in payload.get("units", [])}


def _original_text(text: str) -> str:
    marker = "Original text:"
    start = text.find(marker)
    if start < 0:
        return text.strip()
    body = text[start + len(marker) :]
    end = body.rfind("Source link:")
    if end >= 0:
        body = body[:end]
    return body.strip()


def _sip_mf_relevant(body: str) -> bool:
    text = body.strip()
    if not text or not MF_TOPIC.search(text):
        return False
    mf = len(MF_TOPIC.findall(text))
    stock = len(STOCK_TOPIC.findall(text))
    if stock >= 2 and stock > mf:
        return False
    if stock >= 3 and mf <= 2:
        return False
    if re.search(r"\bstock trading\b", text, re.IGNORECASE) and stock >= mf:
        return False
    if not MF_EXPERIENCE.search(text) and stock >= 1:
        return False
    return True


def _filter_sip_mf_reviews(
    reviews: list[dict],
    question: str = "",
) -> list[dict]:
    portfolio_q = re.search(
        r"\b(portfolio|dashboard|balance|holding|invested amount|folio)\b",
        question,
        re.IGNORECASE,
    )

    def keep(review: dict) -> bool:
        text = review.get("text") or ""
        if _sip_mf_relevant(text):
            return True
        if portfolio_q and re.search(
            r"\b(portfolio|holding|invested|balance|dashboard)\b", text, re.IGNORECASE
        ):
            return True
        return False

    return [review for review in reviews if keep(review)]


def _relevance_boost(
    question: str,
    lexical_q: str,
    unit: dict,
    review: dict,
    score: float,
) -> float:
    text = review.get("text") or ""
    embed = unit.get("embed_text") or text
    boosted = score
    lex = max(lexical_overlap(lexical_q, text), lexical_overlap(lexical_q, embed))
    if lex >= 0.18:
        boosted += min(0.06, (lex - 0.18) * 0.5)
    q_lower = question.lower()
    doc_tokens = token_set(text + " " + embed)
    for term in token_set(lexical_q):
        if len(term) < 5 or term not in doc_tokens:
            continue
        if term in q_lower or term in lexical_q.lower():
            boosted += 0.025
    q_stock = bool(STOCK_TOPIC.search(question) or re.search(r"\bstocks?\b", question, re.I))
    q_mf = bool(MF_TOPIC.search(question))
    if q_stock and q_mf:
        doc_stock = bool(STOCK_TOPIC.search(text) or re.search(r"\bstocks?\b", text, re.I))
        doc_mf = bool(MF_TOPIC.search(text))
        if doc_stock and doc_mf and MF_EXPERIENCE.search(text):
            boosted += 0.07
    return min(1.0, boosted)


def _as_review(hit: dict, units: dict[str, dict]) -> dict:
    parent = _parent_id(hit["id"])
    unit = units.get(parent, {})
    full = unit.get("text") or hit["text"]
    review = dict(hit)
    review["matched_chunk"] = hit["id"]
    review["id"] = parent
    review["text"] = _original_text(full)
    review["reviewer"] = _reviewer(full)
    url = unit.get("citation_url") or review.get("citation_url") or ""
    review["citation_url"] = "" if not url or _is_store_url(url) else url
    if unit.get("source_name"):
        review["source_name"] = unit["source_name"]
    if unit.get("last_updated"):
        review["last_updated"] = unit["last_updated"]
    review["source_type"] = unit.get("source_type") or hit.get("source_type") or ""
    if unit.get("english_translation"):
        review["english_translation"] = unit["english_translation"]
    return review


def _best_by_review(chunk_hits: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for hit in chunk_hits:
        parent = _parent_id(hit["id"])
        current = best.get(parent)
        if current is None or hit["similarity"] > current["similarity"]:
            best[parent] = hit
    return sorted(best.values(), key=lambda hit: hit["similarity"], reverse=True)


def _dot_similarity(query_vector, embedding: list[float]) -> float:
    import numpy as np

    a = np.asarray(query_vector, dtype=np.float32)
    b = np.asarray(embedding, dtype=np.float32)
    return float(np.dot(a, b))


def _merge_lexical_supplements(
    hits: list[dict],
    lexical_q: str,
    units: dict[str, dict],
    collection,
    query_vector,
) -> list[dict]:
    seen = {_parent_id(h["id"]) for h in hits}
    ranked: list[tuple[float, str]] = []
    for pid, unit in units.items():
        doc = unit.get("embed_text") or unit.get("text") or ""
        lex = lexical_overlap(lexical_q, doc)
        if lex < MIN_LEXICAL_SUPPLEMENT or pid in seen:
            continue
        ranked.append((lex, pid))
    ranked.sort(key=lambda row: row[0], reverse=True)
    if not ranked:
        return hits
    chunk_ids: list[str] = []
    for lex, pid in ranked[:LEXICAL_SUPPLEMENT_UNITS]:
        try:
            got = collection.get(where={"parent_id": pid}, include=[])
            ids = got.get("ids") or []
        except Exception:
            ids = []
        chunk_ids.append(ids[0] if ids else f"{pid}-c01")
    got = collection.get(
        ids=chunk_ids,
        include=["documents", "metadatas", "embeddings"],
    )
    for chunk_id, text, meta, embedding in zip(
        got.get("ids") or [],
        got.get("documents") or [],
        got.get("metadatas") or [],
        (got.get("embeddings") if got.get("embeddings") is not None else []) or [],
    ):
        if not chunk_id or not embedding:
            continue
        meta = meta or {}
        vector_sim = _dot_similarity(query_vector, embedding)
        hits.append(
            {
                "id": chunk_id,
                "text": text or "",
                "source_name": meta.get("source_name") or "",
                "section": meta.get("section") or "",
                "citation_url": meta.get("citation_url") or "",
                "last_updated": meta.get("last_updated") or "",
                "source_type": meta.get("source_type") or "",
                "parent_id": meta.get("parent_id") or "",
                "similarity": vector_sim,
            }
        )
    return hits


def retrieve(
    question: str,
    k: int = TOP_K,
    *,
    lexical_question: str | None = None,
) -> list[dict]:
    import numpy as np

    global last_hits

    expanded = expand_query(question)
    lexical_q = expand_query(lexical_question or question)
    vector = get_model().encode(
        [expanded],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    collection = get_client().get_collection(COLLECTION_NAME)
    stored_model = (collection.metadata or {}).get("model_id")
    if stored_model and stored_model != MODEL_ID:
        raise RuntimeError(
            f"Vector store model is {stored_model}; query model is {MODEL_ID}."
        )
    query = np.asarray(vector, dtype=np.float32)
    units = _units_by_id()
    total_chunks = collection.count()
    n_results = min(total_chunks, max(GLOBAL_CHUNK_CANDIDATES, total_chunks))
    result = collection.query(
        query_embeddings=[query.tolist()],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    hits = _hits_from_result(result)
    hits = _merge_lexical_supplements(hits, lexical_q, units, collection, query)
    for hit in hits:
        parent = _parent_id(hit["id"])
        unit = units.get(parent, {})
        doc = unit.get("embed_text") or unit.get("text") or hit["text"]
        lex = max(
            lexical_overlap(lexical_q, doc),
            lexical_overlap(lexical_q, hit["text"]),
        )
        vector_sim = hit["similarity"]
        hit["vector_similarity"] = vector_sim
        hit["similarity"] = hybrid_score(vector_sim, lex)
    by_review = _best_by_review(hits)
    reviews: list[dict] = []
    for hit in by_review:
        if hit["similarity"] < MIN_SIMILARITY:
            continue
        review = _as_review(hit, units)
        unit = units.get(review["id"], {})
        overlap = max(
            lexical_overlap(lexical_q, review["text"]),
            lexical_overlap(lexical_q, unit.get("embed_text") or ""),
        )
        vector_sim = hit.get("vector_similarity", hit["similarity"])
        if overlap < 0.06 and vector_sim < 0.42:
            continue
        review["similarity"] = _relevance_boost(
            lexical_question or question,
            lexical_q,
            unit,
            review,
            review["similarity"],
        )
        reviews.append(review)
    reviews = _filter_sip_mf_reviews(
        reviews,
        lexical_question or question,
    )
    if reviews and not question_supported_by_texts(
        lexical_question or question,
        [row["text"] for row in reviews],
    ):
        last_hits = []
        return []
    reviews.sort(key=lambda row: row["similarity"], reverse=True)
    review_by_id: dict[str, dict] = {row["id"]: row for row in reviews}
    vector_pool_ids = set(review_by_id)
    user_q = lexical_question or question
    bm25_index = get_bm25_index()
    bm25_a = bm25_index.normalized_scores(lexical_q, top_k=120)
    bm25_b = bm25_index.normalized_scores(lexical_focus_query(lexical_q), top_k=120)
    bm25_scores = dict(bm25_a)
    for pid, score in bm25_b.items():
        bm25_scores[pid] = max(bm25_scores.get(pid, 0.0), score)

    def _inject_record(pid: str, hybrid_seed: float) -> None:
        if pid in review_by_id or pid not in units:
            return
        fake_hit = {
            "id": f"{pid}-c01",
            "similarity": hybrid_seed,
            "vector_similarity": hybrid_seed,
            "text": "",
        }
        review_by_id[pid] = _as_review(fake_hit, units)

    for pid, bm25_norm in bm25_scores.items():
        unit = units.get(pid)
        if not unit:
            continue
        doc = unit.get("embed_text") or unit.get("text") or ""
        facet = facet_alignment_score(user_q, doc)
        spec = document_specific_boost(user_q, doc)
        if pid in vector_pool_ids:
            continue
        if bm25_norm < 0.38 and spec < 0.08 and bm25_norm < 0.5:
            continue
        if not facet_overlap_required(user_q, doc) and spec < 0.08 and bm25_norm < 0.48:
            continue
        lex = lexical_overlap(lexical_q, doc)
        if bm25_norm < 0.45 and lex < 0.07 and spec < 0.12:
            continue
        seed = max(MIN_SIMILARITY, min(0.38, 0.14 + bm25_norm * 0.22 + spec))
        _inject_record(pid, seed)

    portfolio_problem = bool(
        re.search(
            r"\b(portfolio|dashboard|holding|balance|tracking|invested)\b",
            user_q,
            re.IGNORECASE,
        )
        and PROBLEM_QUERY.search(user_q)
    )
    if portfolio_problem:
        candidates: list[tuple[float, str]] = []
        for pid, unit in units.items():
            if pid in review_by_id:
                continue
            doc = unit.get("embed_text") or unit.get("text") or ""
            pms = portfolio_monitoring_score(user_q, doc)
            if pms < 0.16:
                continue
            if TRACKING_QUERY.search(user_q) and not MONITORING_SIGNAL.search(doc):
                continue
            candidates.append((pms, pid))
        candidates.sort(key=lambda row: row[0], reverse=True)
        for pms, pid in candidates[:12]:
            bm25_norm = bm25_scores.get(pid) or bm25_index.normalized_score_for(
                lexical_q, pid
            )
            seed = max(
                MIN_SIMILARITY,
                min(0.44, 0.18 + pms + bm25_norm * 0.1),
            )
            _inject_record(pid, seed)

    if SIP_STOCK_QUERY.search(user_q):
        journey_candidates: list[tuple[float, str]] = []
        for pid, unit in units.items():
            if pid in review_by_id:
                continue
            doc = unit.get("embed_text") or unit.get("text") or ""
            score = sip_stock_journey_alignment(user_q, doc)
            if score < 0.14:
                continue
            journey_candidates.append((score, pid))
        journey_candidates.sort(key=lambda row: row[0], reverse=True)
        for score, pid in journey_candidates[:5]:
            bm25_norm = bm25_scores.get(pid) or bm25_index.normalized_score_for(
                lexical_q, pid
            )
            seed = max(MIN_SIMILARITY, min(0.4, 0.15 + score + bm25_norm * 0.1))
            _inject_record(pid, seed)

    support_focus = bool(
        SUPPORT_QUERY.search(user_q) or SUPPORT_QUALITY_QUERY.search(user_q)
    )
    if support_focus:
        support_candidates: list[tuple[float, str]] = []
        for pid, unit in units.items():
            if pid in review_by_id:
                continue
            doc = unit.get("embed_text") or unit.get("text") or ""
            sea = support_experience_alignment(user_q, doc)
            if sea < 0.16:
                continue
            support_candidates.append((sea, pid))
        support_candidates.sort(key=lambda row: row[0], reverse=True)
        for sea, pid in support_candidates[:10]:
            bm25_norm = bm25_scores.get(pid) or bm25_index.normalized_score_for(
                lexical_q, pid
            )
            seed = max(MIN_SIMILARITY, min(0.42, 0.16 + sea + bm25_norm * 0.08))
            _inject_record(pid, seed)

    if re.search(
        r"\b(why|expect|expectation|motivation|reason)\b.{0,80}\b(start|sip)\b",
        user_q,
        re.IGNORECASE | re.DOTALL,
    ) or re.search(
        r"\bwhy do\b.{0,40}\b(?:investors|people|folks)\b",
        user_q,
        re.IGNORECASE,
    ):
        mot_candidates: list[tuple[float, str]] = []
        for pid, unit in units.items():
            if pid in review_by_id:
                continue
            doc = unit.get("embed_text") or unit.get("text") or ""
            mea = motivation_expectation_alignment(user_q, doc)
            if mea < 0.14:
                continue
            mot_candidates.append((mea, pid))
        mot_candidates.sort(key=lambda row: row[0], reverse=True)
        for mea, pid in mot_candidates[:8]:
            bm25_norm = bm25_scores.get(pid) or bm25_index.normalized_score_for(
                lexical_q, pid
            )
            seed = max(MIN_SIMILARITY, min(0.42, 0.16 + mea + bm25_norm * 0.1))
            _inject_record(pid, seed)

    rescored: list[dict] = []
    for pid, review in review_by_id.items():
        unit = units.get(pid, {})
        doc = unit.get("embed_text") or review.get("text") or ""
        hybrid = float(review.get("similarity", 0.0))
        bm25_norm = bm25_scores.get(pid) or bm25_index.normalized_score_for(
            lexical_q, pid
        )
        facet = facet_alignment_score(user_q, doc)
        from_pool = pid in vector_pool_ids
        rel = combined_rerank_score(
            hybrid,
            bm25_norm,
            facet,
            user_q,
            doc,
            from_vector_pool=from_pool,
        )
        review = dict(review)
        review["bm25_norm"] = bm25_norm
        review["facet_score"] = facet
        review["relevance_score"] = rel
        review["from_vector_pool"] = from_pool
        rescored.append(review)

    rescored = _filter_sip_mf_reviews(rescored, user_q)
    kept: list[dict] = []
    for review in rescored:
        rel = review.get("relevance_score", 0.0)
        hybrid = review.get("similarity", 0.0)
        from_pool = review.get("from_vector_pool", True)
        row = dict(review)
        if from_pool and hybrid >= MIN_SIMILARITY and rel >= 0.18:
            kept.append(row)
        elif rel >= 0.27:
            kept.append(row)
        elif not from_pool and rel >= 0.28 and (
            review.get("facet_score", 0) >= 0.12
            or transaction_status_alignment(user_q, review.get("text") or "") >= 0.12
        ):
            kept.append(row)

    if kept and not question_supported_by_texts(user_q, [row["text"] for row in kept]):
        last_hits = []
        return []

    kept.sort(
        key=lambda row: (
            row.get("relevance_score", row.get("similarity", 0.0)),
            row.get("similarity", 0.0),
            row.get("bm25_norm", 0.0),
        ),
        reverse=True,
    )
    reviews = kept[: max(k, MAX_RECORDS)]
    last_hits = reviews
    return reviews


def _citation(hit: dict) -> str:
    if hit.get("citation_url"):
        return hit["citation_url"]
    parts = [hit.get("source_name") or "corpus"]
    if hit.get("section"):
        parts.append(hit["section"])
    return " — ".join(parts)


def _three_sentences(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p for p in parts if p]
    return " ".join(parts[:3])


METRIC = (
    "The business goal is to increase the share of first-time mutual fund investors "
    "who still have an active SIP 12 months after their first one. "
    "Active means at least one successful instalment in the last 30 days."
)
PROOF = re.compile(
    r"\b(?:show|give|share|list)\b.{0,20}\b(?:source|sources|proof|proofs|reviews)\b"
    r"|^(?:source|sources|proof|proofs)\b",
    re.IGNORECASE,
)
GENERIC_TITLE = re.compile(
    r"^(?:google\s+)?play store review|app store review$",
    re.IGNORECASE,
)


def _is_store_url(url: str) -> bool:
    lowered = url.lower()
    return (
        "play.google.com" in lowered
        or "itunes.apple.com" in lowered
        or "apps.apple.com" in lowered
    )


def _reviewer(text: str) -> str:
    first = text.splitlines()[0].strip() if text else ""
    match = re.match(r"Record\s+\d+\s*:\s*(.*)", first, re.IGNORECASE)
    title = match.group(1).strip() if match else ""
    if not title or GENERIC_TITLE.match(title):
        if "app store" in first.lower() and "play" not in first.lower():
            return "App Store review"
        return "Play Store review"
    return title


def _select_evidence_for_answer(
    hits: list[dict],
    lexical_q: str,
    *,
    limit: int = 10,
) -> list[dict]:
    if len(hits) <= limit:
        return hits
    ranked = sorted(
        hits,
        key=lambda h: (
            h["similarity"],
            lexical_overlap(lexical_q, h.get("text") or ""),
        ),
        reverse=True,
    )
    return ranked[:limit]


def _corpus_ingested_on() -> str:
    if not LOADED_PATH.is_file():
        return "unknown"
    payload = json.loads(LOADED_PATH.read_text(encoding="utf-8"))
    return payload.get("ingested_on") or "unknown"


def _prompt(question: str, hits: list[dict]) -> str:
    items = [to_evidence_item(h) for h in hits]
    evidence = prompt_block(items)
    return (
        "You answer only from the records below. Do not use outside knowledge.\n"
        f"{METRIC}\n"
        "Each block is one source record. voice=advice_or_commenter means the text is "
        "guidance to someone else, not necessarily the author's own experience.\n"
        "Summarise what the records say about the question. Report individual examples when "
        "only one record mentions them.\n"
        "Separate personal experiences, commenter advice, multi-record patterns, and any "
        "brief inference (label inferences explicitly).\n"
        "Do not infer internal product bugs — only describe what was reported.\n"
        "Do not give investment advice. Do not use a table.\n"
        "End each bullet or sentence that describes a specific record's experience with that "
        "record id in square brackets, e.g. [rec-0192]. Use only ids from the records below.\n"
        "If the records only partly address the question, summarise the supported part and note "
        "which part has limited evidence. Do not reject the whole question when some records "
        "are relevant.\n"
        "Use the exact phrase \"The documents do not cover this.\" only when none of the supplied "
        "records relate to the question.\n\n"
        "Use exactly these markdown headings:\n"
        "## What the reviews say\n"
        "## Summary\n\n"
        f"Question: {question}\n\n"
        f"Records ({len(hits)} unique source records supplied to you):\n{evidence}"
    )


def _normalize_body(body: str) -> str:
    text = body.strip()
    if not text:
        return text
    if "## What the reviews say" not in text:
        if text.lower().startswith("the documents do not cover this"):
            return text
        text = "## What the reviews say\n" + text
    if "## Summary" not in text:
        parts = text.split("## What the reviews say", 1)
        if len(parts) == 2:
            main = parts[1].strip()
            split_at = main.find("\n\n")
            if split_at > 0:
                say = main[:split_at].strip()
                rest = main[split_at:].strip()
                text = f"## What the reviews say\n{say}\n\n## Summary\n{rest or say}"
            else:
                text = f"## What the reviews say\n{main}\n\n## Summary\n{main}"
    return text


def _format_source_links(hits: list[dict], max_links: int = 3) -> str:
    lines: list[str] = []
    seen_records: set[str] = set()
    seen_urls: set[str] = set()
    for hit in hits:
        rid = hit["id"]
        if rid in seen_records:
            continue
        seen_records.add(rid)
        source = hit.get("source_type") or "source"
        title = hit.get("reviewer") or rid
        url = hit.get("citation_url") or ""
        if url:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            lines.append(f"**{rid}** · {source} · {title}\n{url}")
        else:
            excerpt = " ".join((hit.get("text") or "").split())
            if len(excerpt) > 360:
                excerpt = excerpt[:360].rsplit(" ", 1)[0] + "…"
            lines.append(f"**{rid}** · {source} · {title}\n{excerpt}")
        if len(lines) >= max_links:
            break
    if not lines and hits:
        hit = hits[0]
        excerpt = " ".join(hit["text"].split())[:360]
        lines.append(
            f"**{hit['id']}** · {hit.get('source_type') or 'source'}\n{excerpt}"
        )
    return "\n\n".join(lines)


def _format_user_answer(
    body: str,
    evidence_hits: list[dict],
    citation_block: str,
) -> str:
    updated = _corpus_ingested_on()
    count = len({hit["id"] for hit in evidence_hits})
    return (
        f"{body.rstrip()}\n\n"
        "## Relevant reviews or source links\n"
        f"{citation_block}\n\n"
        f"**Relevant reviews found:** {count} "
        f"unique source records selected for this answer "
        f"(retrieval over the corpus; not every record was searched exhaustively).\n\n"
        f"Last updated from sources: {updated}"
    )


def _lead_source(hits: list[dict]) -> str:
    specific = [hit for hit in hits if hit.get("citation_url")]
    if specific:
        best = max(specific, key=lambda hit: hit["similarity"])
        return best["citation_url"]
    closest = hits[0]
    name = closest.get("reviewer") or "Review"
    excerpt = " ".join(closest["text"].split())
    if len(excerpt) > 420:
        excerpt = excerpt[:420].rsplit(" ", 1)[0] + "…"
    return f"Closest review ({name}): {excerpt}"


def format_proofs(hits: list[dict]) -> str:
    if not hits:
        return "Ask a question first. I can then show the reviews behind that answer."
    lines = [
        f"**Relevant reviews found:** {len(hits)} "
        "(identified by retrieval; not an exhaustive corpus count.)",
        "",
    ]
    for hit in hits:
        source = hit.get("source_type") or "source"
        lines.append(f"**{hit['id']}** · {source} · {hit.get('reviewer') or 'Review'}")
        if hit.get("citation_url"):
            lines.append(hit["citation_url"])
        else:
            text = hit["text"].strip()
            if hit.get("english_translation"):
                text = f"{text}\nEnglish: {hit['english_translation']}"
            lines.append(text)
        lines.append("")
    return "\n".join(lines).strip()


def _generate(prompt: str) -> str:
    from groq_client import generate

    return generate(prompt)


def _top_relevance(hit: dict) -> float:
    return float(
        hit.get("relevance_score", hit.get("similarity", 0.0)),
    )


def answer(question: str, history: list[dict] | None = None) -> str:
    global last_hits
    greeted = greeting_reply(question)
    if greeted:
        return greeted
    blocked = guard(question)
    if blocked:
        return blocked
    if PROOF.search(question.strip()) and last_hits:
        return format_proofs(last_hits)
    search_question = contextualize_question(question, history)
    hits = retrieve(search_question, lexical_question=question)
    if not hits or _top_relevance(hits[0]) < MIN_SIMILARITY:
        return "The documents do not cover this."
    lexical_q = expand_query(question)
    evidence_hits = select_evidence(hits, lexical_q)
    items = [to_evidence_item(h) for h in evidence_hits]
    last_hits[:] = evidence_hits
    prompt = _prompt(question, evidence_hits)
    body = _generate(prompt).strip()
    if not body or body.lower().startswith("the documents do not cover this"):
        return "The documents do not cover this."
    if "## Summary" not in body or "## What the reviews say" not in body:
        body = _generate(
            prompt + "\n\nYou must include both markdown headings exactly as specified."
        ).strip()
    body = _normalize_body(body)
    if body.lower().startswith("the documents do not cover this"):
        return "The documents do not cover this."
    valid_ids = {item["id"] for item in items}
    cited = parse_cited_record_ids(body, valid_ids)
    if not cited:
        cited = fallback_cited_ids(items, lexical_q)
    cited, cite_issues = filter_cited_ids_by_validation(body, items, cited)
    citation_block = format_citations_for_ids(items, cited)
    return _format_user_answer(body, evidence_hits, citation_block)


def main() -> None:
    args = sys.argv[1:]
    chunks_only = False
    if args and args[0] == "--chunks":
        chunks_only = True
        args = args[1:]
    question = " ".join(args).strip()
    if not question:
        print('Usage: python Code/retrieval.py [--chunks] "your question"')
        return
    blocked = guard(question)
    if blocked:
        print(blocked)
        return
    if chunks_only:
        hits = retrieve(question)
        if not hits or hits[0]["similarity"] < MIN_SIMILARITY:
            print("The documents do not cover this.")
            return
        for hit in hits:
            print(f"{hit['id']}  chunk={hit.get('matched_chunk')}  sim={hit['similarity']:.3f}")
            print(hit["text"][:220].replace("\n", " "))
            print(hit["citation_url"])
            print("---")
        return
    print(answer(question))


if __name__ == "__main__":
    main()
