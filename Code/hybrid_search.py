"""Hybrid lexical + semantic helpers for retrieval (no question-specific rules)."""

from __future__ import annotations

import re

TOKEN = re.compile(r"[a-z0-9']+")

# Domain vocabulary expansion only — not tied to any fixed PRD question.
EXPANSIONS: dict[str, tuple[str, ...]] = {
    "sip": ("systematic", "instalment", "installment", "autopay", "mandate"),
    "mutual": ("fund", "mf"),
    "fund": ("mutual", "sip"),
    "pause": ("stop", "cancel", "discontinue", "lapse"),
    "stop": ("pause", "cancel", "discontinue"),
    "start": ("begin", "started", "starting", "first"),
    "motivation": ("reason", "expect", "why", "goal"),
    "expect": ("expectation", "hope", "goal"),
    "confus": ("unclear", "worry", "worried", "uncertain"),
    "withdraw": ("redeem", "redemption", "pull"),
    "autopay": ("autopay", "mandate", "debit", "payment"),
    "customer": ("support", "service", "help"),
    "investor": ("user", "people", "customer", "beginner"),
    "user": ("investor", "people", "customer"),
    "people": ("investor", "user", "customer"),
    "behav": ("change", "changed", "habit", "increase", "reduce", "continue"),
    "change": ("increase", "reduce", "stop", "continue", "habit", "scale"),
    "month": ("months", "year", "years", "later", "timeline", "after"),
    "student": ("college", "pocket", "allowance", "beginner", "job"),
    "income": ("salary", "hike", "amount", "afford", "small"),
    "return": ("growth", "expect", "promised", "outcome", "percent"),
    "stock": ("trading", "equity", "shares", "market"),
    "friend": ("forum", "community", "reddit", "advice", "suggest"),
    "forum": ("reddit", "community", "discussion", "online"),
    "kyc": ("verification", "pan", "documents", "pending"),
    "mandate": ("autopay", "debit", "payment", "bank"),
    "instal": ("installment", "debit", "failed", "deducted"),
    "goal": ("dream", "house", "retirement", "long", "term"),
    "systematic": ("sip", "instalment", "monthly"),
    "dashboard": ("portfolio", "balance", "holding", "display", "value"),
    "portfolio": ("holding", "balance", "invested", "tracking", "dashboard"),
    "beginner": ("newbie", "first", "starting", "student"),
    "support": ("customer", "care", "ticket", "chat", "help", "helpline"),
    "deduct": ("debit", "failed", "autopay", "instalment", "payment"),
    "debited": ("deducted", "debit", "payment", "instalment"),
    "pending": ("status", "processing", "updating", "stuck", "pending"),
    "status": ("pending", "processing", "updating", "payment"),
    "payment": ("instalment", "deduct", "debit", "sip", "status"),
    "responsive": ("helpful", "response", "quick", "resolved"),
    "reach": ("contact", "helpline", "chat", "call", "email"),
    "hard": ("difficult", "unhelpful", "pathetic", "worst"),
    "describe": ("report", "say", "reviews", "experiences"),
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if len(t) > 2]


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    return token


def token_set(text: str) -> set[str]:
    raw = tokenize(text)
    return {t for t in raw} | {_stem(t) for t in raw}


GENERIC_QUERY_TERMS = frozenset(
    {
        "sip",
        "sips",
        "mutual",
        "fund",
        "funds",
        "groww",
        "invest",
        "investing",
        "investment",
        "investor",
        "investors",
        "people",
        "user",
        "users",
        "customer",
        "customers",
        "review",
        "reviews",
        "say",
        "said",
        "what",
        "when",
        "where",
        "why",
        "how",
        "does",
        "do",
        "the",
        "and",
        "for",
        "about",
        "with",
        "from",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those",
        "after",
        "before",
        "into",
        "have",
        "has",
        "been",
        "being",
        "would",
        "could",
        "should",
        "available",
        "documents",
    }
)


def anchor_terms(question: str) -> set[str]:
    return {
        _stem(t)
        for t in token_set(question)
        if t not in GENERIC_QUERY_TERMS and len(t) >= 4
    }


def question_supported_by_texts(question: str, texts: list[str]) -> bool:
    """False when distinctive query terms never appear in candidate evidence."""
    if not texts:
        return False
    corpus = token_set(" ".join(texts))
    if _multi_clause_supported(question, corpus):
        return True
    expanded = expand_query(question)
    anchors = anchor_terms(question) | anchor_terms(expanded)
    if not anchors:
        return True
    matched = anchors & corpus
    subject_anchors = anchor_terms(question) or anchors
    subject_matched = subject_anchors & corpus
    if subject_anchors and not subject_matched:
        return False
    if len(subject_anchors) >= 2:
        ratio = len(subject_matched) / len(subject_anchors)
        if ratio < 0.34:
            return False
    for token in tokenize(question):
        if len(token) >= 11 and token not in GENERIC_QUERY_TERMS:
            stem = _stem(token)
            if token not in corpus and stem not in corpus:
                return False
    return bool(matched) or not subject_anchors


def _multi_clause_supported(question: str, corpus: set[str]) -> bool:
    """Broad questions with multiple facets pass if any clause is grounded in evidence."""
    parts = re.split(r",\s*and\s+|\?\s*|\;\s*", question.strip())
    clauses = [p.strip(" ,?") for p in parts if len(p.strip()) >= 12]
    if len(clauses) < 2:
        return False
    grounded = 0
    for clause in clauses:
        anchors = anchor_terms(clause)
        if not anchors:
            grounded += 1
            continue
        matched = anchors & corpus
        if not matched:
            continue
        if len(anchors) < 2 or len(matched) / len(anchors) >= 0.34:
            grounded += 1
    return grounded >= 1


def expand_query(question: str) -> str:
    tokens = tokenize(question)
    extra: set[str] = set()
    for token in tokens:
        for key, syns in EXPANSIONS.items():
            if key in token or token.startswith(key[:4]):
                extra.update(syns)
    if not extra:
        return question
    return question + " " + " ".join(sorted(extra))


def lexical_focus_query(question: str) -> str:
    """Extra lexical signal from distinctive query terms (not question-specific)."""
    expanded = expand_query(question)
    anchors = sorted(anchor_terms(question) | anchor_terms(expanded))
    focus = [a for a in anchors if len(a) >= 4][:14]
    return expanded + " " + " ".join(focus)


def lexical_overlap(query: str, document: str) -> float:
    qt = token_set(query)
    if not qt:
        return 0.0
    dt = token_set(document)
    return len(qt & dt) / len(qt)


def hybrid_score(vector_similarity: float, lexical: float) -> float:
    lex_c = min(1.0, lexical)
    if lex_c >= 0.20:
        return 0.20 * vector_similarity + 0.80 * lex_c
    if lex_c >= 0.12:
        return 0.55 * vector_similarity + 0.45 * lex_c
    return 0.72 * vector_similarity + 0.28 * lex_c


def contextualize_question(question: str, history: list[dict] | None) -> str:
    text = question.strip()
    if not history:
        return text
    if not re.search(
        r"\b(this|that|these|those|they|them|their|it|same|above|earlier|follow[- ]?up)\b",
        text,
        re.IGNORECASE,
    ):
        return text
    for message in reversed(history):
        if message.get("role") == "user":
            return f"Earlier question: {message['content']}\nFollow-up: {text}"
    return text
