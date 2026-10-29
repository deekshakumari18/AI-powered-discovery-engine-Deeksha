"""Manual-style citation review for known-gap questions (evaluation)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from citation_validate import (
    claim_supported_by_source,
    validate_answer_citations,
)
from evidence import (
    explain_evidence_selection,
    parse_cited_record_ids,
    to_evidence_item,
)
from hybrid_search import expand_query
from retrieval import answer, retrieve


GAP_QUESTIONS = [
    (
        "payment_status_paraphrase",
        "What happens when instalments are debited but payment or SIP status stays pending?",
    ),
    (
        "support_mixed_experiences",
        "Do reviews describe Groww support as responsive or hard to reach?",
    ),
]


def review_question(name: str, question: str, *, generate: bool = True) -> dict:
    lexical_q = expand_query(question)
    hits = retrieve(question, lexical_question=question)
    report = explain_evidence_selection(hits, lexical_q)
    items = [to_evidence_item(h) for h in report["prompt_evidence"]]
    out: dict = {
        "case": name,
        "retrieve_top14": report["retrieval_top14"],
        "prompt_evidence_ids": report["prompt_ids"],
        "dropped_from_top14": report["dropped_from_retrieval_top14"],
        "prompt_tiers": report["tiers"],
    }
    if not generate:
        return out
    text = answer(question)
    valid = {i["id"] for i in items}
    cited = parse_cited_record_ids(text, valid)
    issues = validate_answer_citations(text, items, valid_ids=valid)
    claim_checks = []
    by_id = {i["id"]: i["text"] for i in items}
    for rid in cited:
        snippet = ""
        for line in text.splitlines():
            if rid in line.lower():
                snippet = line[:240]
                break
        ok, overlap = claim_supported_by_source(snippet, by_id.get(rid, ""))
        claim_checks.append(
            {"id": rid, "lexical_ok": ok, "overlap": round(overlap, 4), "snippet": snippet[:120]}
        )
    out["answer_excerpt"] = text[:1200]
    out["cited_ids"] = cited
    out["validator_issues"] = issues
    out["claim_lexical_checks"] = claim_checks
    return out


if __name__ == "__main__":
    generate = "--no-llm" not in sys.argv
    results = [review_question(n, q, generate=generate) for n, q in GAP_QUESTIONS]
    print(json.dumps(results, indent=2))
