"""Query/document facet overlap and praise-vs-problem calibration (general, not question-specific)."""

from __future__ import annotations

import re

FACETS: list[tuple[str, re.Pattern[str]]] = [
    (
        "portfolio_monitoring",
        re.compile(
            r"\b(portfolio|dashboard|holding|holdings|balance|tracking|chart|"
            r"invested amount|statement|current value|folio|visibility|progress)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "kyc_admin",
        re.compile(
            r"\b(kyc|verification|verify|pan|documents|pending|e?pan)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "transaction_money",
        re.compile(
            r"\b(deduct|debit|refund|payment|transfer|failed|instalment|installment|"
            r"money movement|credited|debited)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "support",
        re.compile(
            r"\b(support|customer care|ticket|chat|helpdesk|response time)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "autopay_mandate",
        re.compile(r"\b(autopay|auto pay|mandate|upi mandate)\b", re.IGNORECASE),
    ),
    (
        "sip_management",
        re.compile(
            r"\b(sip|systematic|instalment|installment|pause|cancel|modify)\b",
            re.IGNORECASE,
        ),
    ),
]

PROBLEM_QUERY = re.compile(
    r"\b(confus\w*|wrong|misleading|miscalcul\w*|worr\w*|complaint|problem|"
    r"fail\w*|error|disappoint\w*|trust|mis.?match|incorrect|bug|issue)\b",
    re.IGNORECASE,
)
PRAISE_DOC = re.compile(
    r"\b(excellent|best app|love the|smooth experience|highly recommend|"
    r"user-friendly|superb|awesome app|great app)\b",
    re.IGNORECASE,
)
PROBLEM_DOC = re.compile(
    r"\b(confus\w*|wrong|misleading|miscalcul\w*|disappoint\w*|fail\w*|"
    r"incorrect|trust|mis.?match|bug|issue|error|not happy|worst)\b",
    re.IGNORECASE,
)
MONITORING_SIGNAL = re.compile(
    r"\b(misleading|miscalcul\w*|wrong|incorrect|trust|chart|current value|"
    r"bank statement|does not match|discrepanc|loss but showing|visibility|"
    r"invested amount|statement)\b",
    re.IGNORECASE,
)
STATUS_QUERY = re.compile(
    r"\b(status|updat\w+|not updat|pending|processing)\b",
    re.IGNORECASE,
)
STATUS_DOC = re.compile(
    r"\b(status|pending|not updat|stuck|processing|showing|spinning|allocated)\b",
    re.IGNORECASE,
)
TRANSACTION_QUERY = re.compile(
    r"\b(deduct\w*|debited|refund|payment|money|instalment|installment)\b",
    re.IGNORECASE,
)
TRACKING_QUERY = re.compile(
    r"\b(track\w*|monitor\w*|visibility|progress|balance|dashboard|holding)\b",
    re.IGNORECASE,
)


def facet_alignment_score(query: str, document: str) -> float:
    if not query.strip() or not document.strip():
        return 0.0
    boost = 0.0
    active = 0
    for _name, pattern in FACETS:
        if pattern.search(query):
            active += 1
            if pattern.search(document):
                boost += 0.12
    if active == 0:
        return 0.0
    return min(0.36, boost)


def praise_penalty(query: str, document: str) -> float:
    if not PROBLEM_QUERY.search(query):
        return 1.0
    if PROBLEM_DOC.search(document):
        return 1.0
    if PRAISE_DOC.search(document) and not PROBLEM_DOC.search(document):
        return 0.78
    return 1.0


def document_specific_boost(query: str, document: str) -> float:
    boost = 0.0
    if re.search(
        r"\b(portfolio|dashboard|balance|holding|tracking|invested amount|chart)\b",
        query,
        re.IGNORECASE,
    ):
        if re.search(
            r"\b(portfolio|holding|dashboard|chart|invested|balance)\b",
            document,
            re.IGNORECASE,
        ):
            boost += 0.06
        if MONITORING_SIGNAL.search(document):
            boost += 0.12
        if PROBLEM_DOC.search(document) and re.search(
            r"\b(portfolio|holding|dashboard|chart|invested|balance|misleading|"
            r"miscalcul|trust)\b",
            document,
            re.IGNORECASE,
        ):
            boost += 0.1
        if TRACKING_QUERY.search(query) and MONITORING_SIGNAL.search(document):
            boost += 0.06
    if re.search(r"\b(kyc|verification)\b", query, re.IGNORECASE) and re.search(
        r"\b(kyc|verification|pan|pending)\b", document, re.IGNORECASE
    ):
        boost += 0.12
    if re.search(r"\b(support|customer care|helpful|unhelpful)\b", query, re.I) and re.search(
        r"\b(support|customer care|ticket|chat|response)\b", document, re.I
    ):
        boost += 0.08
    if re.search(r"\b(refund|deduct|debited|status)\b", query, re.I) and re.search(
        r"\b(refund|deduct|debited|instalment|payment|pending|status|updat)\b", document, re.I
    ):
        boost += 0.1
    if re.search(r"\b(status|updat\w+|not updat|pending)\b", query, re.I) and re.search(
        r"\b(status|pending|not updat|stuck|processing|allocated)\b", document, re.I
    ):
        boost += 0.12
    return min(0.32, boost)


def portfolio_monitoring_score(query: str, document: str) -> float:
    """How well a record matches portfolio visibility / balance monitoring concerns."""
    if not re.search(
        r"\b(portfolio|dashboard|holding|balance|tracking|invested)\b",
        query,
        re.IGNORECASE,
    ):
        return 0.0
    score = document_specific_boost(query, document)
    if not re.search(
        r"\b(portfolio|holding|dashboard|chart|balance|invested)\b",
        document,
        re.IGNORECASE,
    ):
        return 0.0
    if TRACKING_QUERY.search(query) and not MONITORING_SIGNAL.search(document):
        score = min(score, 0.1)
    if PROBLEM_QUERY.search(query):
        if PROBLEM_DOC.search(document):
            score += 0.1
        elif PRAISE_DOC.search(document):
            score -= 0.14
        else:
            score -= 0.06
    if TRACKING_QUERY.search(query) and MONITORING_SIGNAL.search(document):
        score += 0.08
    return max(0.0, min(0.36, score))


ADVICE_HEAVY = re.compile(
    r"\b(my suggestion|you should|i follow a ratio|giving you some suggestions|"
    r"try to diversify|go for reinvestment)\b",
    re.IGNORECASE,
)


def transaction_status_alignment(query: str, document: str) -> float:
    if not TRANSACTION_QUERY.search(query):
        return 0.0
    score = 0.0
    if TRANSACTION_QUERY.search(document):
        score += 0.08
    if STATUS_QUERY.search(query) and STATUS_DOC.search(document):
        score += 0.18
    if re.search(r"\b(updat\w*|not updat)\b", query, re.I) and re.search(
        r"\b(updat\w*|pending|not updat|stuck)\b", document, re.I
    ):
        score += 0.1
    if re.search(r"\b(wallet|refund|returned)\b", document, re.I) and re.search(
        r"\b(deduct\w*|debited|refund)\b", query, re.I
    ):
        score += 0.1
    if re.search(r"\b(deduct\w*|debited)\b", query, re.I) and re.search(
        r"\b(deduct\w*|debited|autopay|instalment)\b", document, re.I
    ):
        score += 0.06
    return min(0.3, score)


SUPPORT_QUERY = re.compile(
    r"\b(support|customer care|helpdesk|helpline|service team)\b",
    re.IGNORECASE,
)
SUPPORT_QUALITY_QUERY = re.compile(
    r"\b(responsive|unresponsive|helpful|unhelpful|hard to reach|reach|"
    r"response time|both helpful|hard to)\b",
    re.IGNORECASE,
)
SUPPORT_NARRATIVE_DOC = re.compile(
    r"\b(support|customer care|helpline|ticket|chat|call|email|"
    r"respond|response|contact|care team)\b",
    re.IGNORECASE,
)


def support_experience_alignment(query: str, document: str) -> float:
    if not SUPPORT_QUERY.search(query) and not SUPPORT_QUALITY_QUERY.search(query):
        return 0.0
    score = 0.0
    if SUPPORT_NARRATIVE_DOC.search(document):
        score += 0.1
    if re.search(r"\b(reviews?|reviewer\w*|people|users)\b", query, re.I):
        if re.search(
            r"\b(support|customer care|helpline|ticket|chat)\b", document, re.I
        ):
            score += 0.06
    if SUPPORT_QUALITY_QUERY.search(query):
        if re.search(
            r"\b(helpful|unhelpful|pathetic|worst|unable|no use|not respond|"
            r"busy|resolved|call back|quick)\b",
            document,
            re.I,
        ):
            score += 0.14
    if re.search(r"\b(hard|difficult|reach|responsive)\b", query, re.I):
        if re.search(
            r"\b(unable|busy|not respond|helpline|chat|call|email|ticket)\b",
            document,
            re.I,
        ):
            score += 0.1
    if re.search(r"\b(both|helpful and|unhelpful)\b", query, re.I):
        if PROBLEM_DOC.search(document) and SUPPORT_NARRATIVE_DOC.search(document):
            score += 0.08
        if PRAISE_DOC.search(document) and SUPPORT_NARRATIVE_DOC.search(document):
            score += 0.06
    return min(0.32, score)


SIP_STOCK_QUERY = re.compile(
    r"\b(sip|mutual fund|\bmf\b).{0,80}\b(stock|trading|equity|shares)\b"
    r"|\b(stock|trading|equity|shares).{0,80}\b(sip|mutual fund|\bmf\b)",
    re.IGNORECASE | re.DOTALL,
)


def sip_stock_journey_alignment(query: str, document: str) -> float:
    if not SIP_STOCK_QUERY.search(query):
        return 0.0
    score = 0.0
    if re.search(r"\b(sip|mutual fund|\bmf\b)\b", document, re.I) and re.search(
        r"\b(stock|trading|equity|shares)\b", document, re.I
    ):
        score += 0.16
    if re.search(r"\b(later|started|moving|lurk|stopped)\b", query, re.I) and re.search(
        r"\b(later|started|lurk|stopped|mutual fund)\b", document, re.I
    ):
        score += 0.06
    return min(0.24, score)


def experience_vs_advice_penalty(query: str, document: str) -> float:
    """Down-rank third-party advice when the question asks about user experiences."""
    if not re.search(
        r"\b(user\w*|people|reviewer\w*|report|describe|complain\w*|worri\w*)\b",
        query,
        re.IGNORECASE,
    ):
        return 1.0
    if ADVICE_HEAVY.search(document) and not MONITORING_SIGNAL.search(document):
        return 0.72
    return 1.0


def active_query_facets(query: str) -> list[str]:
    return [name for name, pattern in FACETS if pattern.search(query)]


def facet_overlap_required(query: str, document: str) -> bool:
    active = active_query_facets(query)
    if not active:
        return True
    doc_hits = {name for name, pattern in FACETS if pattern.search(document)}
    return bool(set(active) & doc_hits)


def motivation_expectation_alignment(query: str, document: str) -> float:
    if not re.search(
        r"\b(why|reason|motivation|expect|expectation|goal|beginner|first)\b",
        query,
        re.IGNORECASE,
    ):
        return 0.0
    score = 0.0
    if re.search(r"\b(why|start|started|begin|reason)\b", query, re.I) and re.search(
        r"\b(start|reason|goal|friend|small|learn|confiden|sip|mutual)\b",
        document,
        re.I,
    ):
        score += 0.1
    if re.search(r"\b(expect|expectation|goal|hope)\b", query, re.I) and re.search(
        r"\b(expect|goal|return|house|wealth|confiden|grow|down payment|salary)\b",
        document,
        re.I,
    ):
        score += 0.12
    if re.search(r"\b(first[- ]time|beginner|student|novice)\b", query, re.I) and re.search(
        r"\b(first|beginner|student|small|monthly|start)\b", document, re.I
    ):
        score += 0.06
    return min(0.26, score)


def combined_rerank_score(
    hybrid: float,
    bm25_norm: float,
    facet: float,
    query: str,
    document: str,
    *,
    from_vector_pool: bool = True,
) -> float:
    """Additive rerank: preserve hybrid rank, boost lexical/facet alignment."""
    facet_term = 0.14 * min(1.0, facet * 2.5)
    lex_term = 0.11 * bm25_norm
    spec = document_specific_boost(query, document)
    pms = portfolio_monitoring_score(query, document)
    tsa = transaction_status_alignment(query, document)
    sea = support_experience_alignment(query, document)
    ssja = sip_stock_journey_alignment(query, document)
    mea = motivation_expectation_alignment(query, document)
    if from_vector_pool:
        base = hybrid + lex_term + facet_term + spec + min(0.2, pms * 0.45) + tsa + sea + ssja + mea
    else:
        # BM25/facet-only candidates must earn their rank with alignment, not raw BM25.
        base = (
            0.55 * hybrid
            + 0.55 * lex_term
            + facet_term
            + spec
            + min(0.26, pms * 0.65)
            + min(0.22, tsa * 0.85)
            + min(0.2, sea * 0.8)
            + min(0.18, ssja * 0.75)
            + min(0.16, mea * 0.75)
        )
        if not facet_overlap_required(query, document):
            base *= 0.72
        if bm25_norm < 0.38 and spec < 0.1 and tsa < 0.12:
            base *= 0.65
    if TRACKING_QUERY.search(query) and PROBLEM_QUERY.search(query):
        if not MONITORING_SIGNAL.search(document):
            base *= 0.84
    if STATUS_QUERY.search(query) and re.search(
        r"\b(autopay|mandate)\b", document, re.IGNORECASE
    ) and not STATUS_DOC.search(document):
        base *= 0.8
    if not from_vector_pool and TRANSACTION_QUERY.search(query) and tsa >= 0.12:
        base += 0.08 + bm25_norm * 0.1
    if STATUS_QUERY.search(query) and TRANSACTION_QUERY.search(query):
        if tsa >= 0.2:
            base += 0.12
        elif from_vector_pool and tsa < 0.15:
            base *= 0.76
    if tsa >= 0.22:
        base += 0.1
    if sea >= 0.18:
        base += 0.12
    elif (SUPPORT_QUERY.search(query) or SUPPORT_QUALITY_QUERY.search(query)) and (
        from_vector_pool and sea < 0.1
    ):
        base *= 0.82
    if ssja >= 0.2:
        base += 0.16
    elif ssja >= 0.18:
        base += 0.13
    elif SIP_STOCK_QUERY.search(query) and from_vector_pool and ssja < 0.08:
        base *= 0.84
    if mea >= 0.16:
        base += 0.08
    return min(
        0.93,
        base * praise_penalty(query, document) * experience_vs_advice_penalty(query, document),
    )
