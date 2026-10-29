"""Phase 6 — Backend evaluation (retrieval + optional answer checks).

Evaluation cases are tests only; they must not be wired into production retrieval.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_search import contextualize_question, question_supported_by_texts
from retrieval import MIN_SIMILARITY, answer, guard, retrieve

LOADED_PATH = PROJECT_ROOT / "Data" / "raw" / "loaded.json"

# expect_any_of: at least one listed parent_id should appear in retrieved hits (top MAX_RECORDS).
# forbid_ids: must not appear. max_hits=0: off-topic / no evidence.
CASES: list[dict] = [
    {
        "name": "motivation_goals_wording_a",
        "question": "Why do first-time investors start a SIP, and what do they expect?",
        "expect_any_of": ["rec-0001", "rec-0003", "rec-0004"],
        "min_hits": 2,
    },
    {
        "name": "motivation_paraphrase_indirect",
        "question": "Why do folks begin systematic investment plans for long-term goals?",
        "expect_any_of": ["rec-0001", "rec-0003", "rec-0004", "rec-0007", "rec-0016"],
        "min_hits": 2,
    },
    {
        "name": "return_expectations",
        "question": "What return or growth outcomes did people hope for when they chose SIP investing?",
        "expect_any_of": ["rec-0221", "rec-0001"],
        "min_hits": 2,
    },
    {
        "name": "student_limited_income",
        "question": "How do students with small amounts describe starting or increasing a SIP?",
        "expect_any_of": ["rec-0002", "rec-0004", "rec-0020"],
        "min_hits": 2,
    },
    {
        "name": "salaried_capacity_change",
        "question": "How do people describe increasing SIP amounts after a salary hike or better income?",
        "expect_any_of": ["rec-0221", "rec-0002", "rec-0004"],
        "min_hits": 2,
    },
    {
        "name": "behaviour_after_losses_or_slow_progress",
        "question": "What changes in behaviour do people report after disappointing returns or slower progress on SIPs?",
        "expect_any_of": ["rec-0221", "rec-0001", "rec-0003"],
        "min_hits": 2,
    },
    {
        "name": "behaviour_months_after_start",
        "question": "What changes in the behaviour of people investing in SIPs, in the months after starting?",
        "expect_any_of": ["rec-0323", "rec-0241", "rec-0095", "rec-0221", "rec-0242"],
        "min_hits": 2,
    },
    {
        "name": "pause_stop_cancel",
        "question": "What leads people to pause, reduce, or stop a SIP?",
        "expect_any_of": ["rec-0221", "rec-0321", "rec-0285"],
        "min_hits": 2,
    },
    {
        "name": "restart_change_amount",
        "question": "Why do people restart or change the amount on an existing SIP?",
        "expect_any_of": ["rec-0323", "rec-0321", "rec-0233"],
        "min_hits": 2,
    },
    {
        "name": "confusion_kyc_mf",
        "question": "What problems do people report with KYC blocking mutual fund or SIP setup?",
        "expect_any_of": ["rec-0032", "rec-0024"],
        "min_hits": 2,
    },
    {
        "name": "mandate_autopay_failed",
        "question": "What issues do people report with autopay mandates or failed SIP instalments?",
        "expect_any_of": ["rec-0321", "rec-0285", "rec-0310"],
        "min_hits": 2,
    },
    {
        "name": "portfolio_invested_amount_confusion",
        "question": "What confusion do users report about how much they invested or their portfolio status?",
        "expect_any_of": ["rec-0192", "rec-0001", "rec-0204"],
        "min_hits": 2,
    },
    {
        "name": "outside_app_help",
        "question": "When unsure about investing, where do people turn besides the app—friends, forums, or support?",
        "expect_any_of": ["rec-0084", "rec-0047", "rec-0222"],
        "min_hits": 2,
    },
    {
        "name": "mixed_sip_and_stocks",
        "question": "Where do people describe starting a SIP and later moving into stock trading?",
        "expect_any_of": ["rec-0222", "rec-0226"],
        "min_hits": 2,
    },
    {
        "name": "follow_up_pronouns",
        "question": "What do they expect after that?",
        "history": [{"role": "user", "content": "Why do first-time investors start a SIP?"}],
        "expect_any_of": ["rec-0001", "rec-0004", "rec-0003"],
        "min_hits": 2,
    },
    {
        "name": "stock_primary_excluded",
        "question": "What do reviews say about continuing a mutual fund SIP after starting?",
        "forbid_ids": ["rec-0209"],
        "min_hits": 1,
    },
    {
        "name": "off_topic_crypto",
        "question": "What do reviews say about cryptocurrency staking rewards on Groww?",
        "max_hits": 0,
    },
    {
        "name": "off_topic_unrelated_product",
        "question": "What do reviews say about Groww home loan EMI restructuring?",
        "max_hits": 0,
    },
]


def _resolve_search(case: dict) -> tuple[str, str]:
    history = case.get("history")
    question = case["question"]
    search_q = question
    if history:
        search_q = contextualize_question(question, history)
    return search_q, question


def run_retrieval_checks() -> tuple[list[str], dict[str, float]]:
    failures: list[str] = []
    recall_hits = 0
    recall_total = 0

    for case in CASES:
        search_q, lexical_q = _resolve_search(case)
        hits = retrieve(search_q, lexical_question=lexical_q)
        ids = {hit["id"] for hit in hits}
        min_hits = case.get("min_hits", 0)
        max_hits = case.get("max_hits")

        if max_hits is not None:
            if hits and hits[0]["similarity"] >= MIN_SIMILARITY:
                failures.append(f"{case['name']}: expected no relevant hits, got {len(hits)}")
            continue

        if len(hits) < min_hits:
            failures.append(f"{case['name']}: expected >= {min_hits} hits, got {len(hits)}")
        if hits and hits[0]["similarity"] < MIN_SIMILARITY:
            failures.append(f"{case['name']}: top similarity below threshold")

        expect_any = case.get("expect_any_of") or case.get("expect_ids") or []
        if expect_any:
            recall_total += 1
            if ids.intersection(expect_any):
                recall_hits += 1
            else:
                failures.append(
                    f"{case['name']}: none of {expect_any} in top {len(hits)} "
                    f"(got {sorted(ids)[:6]}…)"
                )

        for rid in case.get("forbid_ids", []):
            if rid in ids:
                failures.append(f"{case['name']}: excluded record {rid} was retrieved")

    metrics = {
        "labeled_recall": recall_hits / recall_total if recall_total else 1.0,
        "labeled_cases": float(recall_total),
    }
    return failures, metrics


def run_metadata_checks() -> list[str]:
    failures: list[str] = []
    payload = json.loads(LOADED_PATH.read_text(encoding="utf-8"))
    units = {u["id"]: u for u in payload.get("units", [])}
    corpus_dates = {u.get("last_updated") for u in payload.get("units", []) if u.get("last_updated")}
    if not corpus_dates:
        failures.append("loaded.json has no last_updated metadata")
        return failures

    hits = retrieve(
        "Why do people start a mutual fund SIP?",
        lexical_question="Why do people start a mutual fund SIP?",
    )
    if not hits:
        failures.append("metadata checks: retrieval returned no hits")
        return failures

    parent_ids = [hit["id"] for hit in hits]
    if len(parent_ids) != len(set(parent_ids)):
        failures.append("retrieved hits are not deduplicated by parent_id")

    for hit in hits:
        matched = hit.get("matched_chunk") or ""
        if matched and not matched.startswith(hit["id"] + "-c"):
            failures.append(f"{hit['id']}: matched_chunk {matched} does not belong to parent")
        lu = hit.get("last_updated") or ""
        if lu and lu not in corpus_dates:
            failures.append(f"{hit['id']}: last_updated {lu!r} not from corpus metadata")
        if lu == date.today().isoformat() and lu not in corpus_dates:
            failures.append("last_updated appears to be today's date, not corpus ingest metadata")

        unit = units.get(hit["id"], {})
        url = hit.get("citation_url") or ""
        expected = unit.get("citation_url") or ""
        if expected and "play.google.com" not in expected.lower() and url != expected:
            if url and url != expected:
                failures.append(f"{hit['id']}: citation_url mismatch with loaded unit")

    texts = [h["text"] for h in hits]
    if not question_supported_by_texts(
        "Why do people start a mutual fund SIP?", texts
    ):
        failures.append("anchor check rejected a straightforward SIP motivation paraphrase")

    return failures


def run_motivation_regression_checks() -> list[str]:
    """Regression for first-time SIP motivation + expectations (false-negative fix)."""
    from evidence import explain_evidence_selection

    failures: list[str] = []
    exact = "Why do first-time investors start a SIP, and what do they expect?"
    paraphrase = "Why do folks begin systematic investment plans, and what outcomes do they hope for?"
    for name, question in (("exact", exact), ("paraphrase", paraphrase)):
        if guard(question):
            failures.append(f"motivation {name}: guard blocked question")
            continue
        hits = retrieve(question, lexical_question=question)
        if not hits:
            failures.append(f"motivation {name}: retrieve returned no hits")
            continue
        from retrieval import _top_relevance, MIN_SIMILARITY

        if _top_relevance(hits[0]) < MIN_SIMILARITY:
            failures.append(f"motivation {name}: top relevance below MIN_SIMILARITY")
        expect = ["rec-0001", "rec-0003", "rec-0004"]
        ids = {h["id"] for h in hits[:14]}
        if not ids.intersection(expect):
            failures.append(
                f"motivation {name}: none of {expect} in top 14 (got {sorted(ids)[:5]}…)"
            )
        lexical_q = __import__("hybrid_search", fromlist=["expand_query"]).expand_query(
            question
        )
        report = explain_evidence_selection(hits, lexical_q)
        if not report["prompt_ids"]:
            failures.append(f"motivation {name}: empty prompt evidence")
        if not ids.intersection(set(report["prompt_ids"])) and ids.intersection(expect):
            failures.append(
                f"motivation {name}: labelled hit in retrieve top14 but not in LLM prompt"
            )
    return failures


def run_answer_checks() -> list[str]:
    failures: list[str] = []
    failures.extend(run_motivation_regression_checks())
    question = "Why do first-time investors start a SIP, and what do they expect?"
    if guard(question):
        failures.append("guard blocked a normal discovery question")
        return failures
    text = answer(question)
    if text.strip().lower().startswith("the documents do not cover this"):
        failures.append("motivation exact question: answer incorrectly declined coverage")
    for heading in (
        "## What the reviews say",
        "## Summary",
        "## Relevant reviews or source links",
        "**Relevant reviews found:**",
        "Last updated from sources:",
    ):
        if heading not in text:
            failures.append(f"answer missing section: {heading}")
    if "not an exhaustive" not in text.lower() and "selected for this answer" not in text.lower():
        failures.append("answer does not clarify retrieval-limited count")
    if re.search(r"\|\s*Record\s*\|", text):
        failures.append("answer used a table layout")
    return failures


RANKING_CASES = [
    {
        "name": "portfolio_dashboard_blind",
        "question": "What worries users about tracking their portfolio after they start a SIP?",
        "expect_any_of": ["rec-0192", "rec-0158", "rec-0190"],
        "top_k": 14,
    },
    {
        "name": "dashboard_vs_beginners_blind",
        "question": "What problems do beginners describe when starting on the app, compared with complaints about wrong balances on the dashboard?",
        "expect_any_of": ["rec-0158", "rec-0192", "rec-0032"],
        "top_k": 14,
    },
    {
        "name": "kyc_mutual_funds",
        "question": "What do people report about failed KYC or verification steps when trying to invest in mutual funds?",
        "expect_any_of": ["rec-0032", "rec-0024"],
        "top_k": 14,
    },
    {
        "name": "mixed_sip_stock_journey",
        "question": "Where do people describe starting a SIP and later moving into stock trading?",
        "expect_any_of": ["rec-0222", "rec-0226"],
        "top_k": 14,
    },
    {
        "name": "transaction_refund_vs_status",
        "question": "What do reviews say about money deducted but SIP or payment status not updating?",
        "expect_any_of": ["rec-0285", "rec-0310", "rec-0321"],
        "top_k": 14,
    },
    {
        "name": "positive_and_negative_same_topic",
        "question": "What do reviews say about Groww customer support — both helpful and unhelpful experiences?",
        "expect_any_of": ["rec-0032", "rec-0047", "rec-0325"],
        "top_k": 14,
    },
    {
        "name": "keyword_trap_irrelevant_praise",
        "question": "What confusion do people report about portfolio values and dashboard balances?",
        "expect_any_of": ["rec-0192", "rec-0158"],
        "forbid_top1_praise_only": True,
        "top_k": 5,
    },
    {
        "name": "portfolio_values_alt_wording",
        "question": "What do reviewers say about misleading portfolio charts or invested amounts on Groww?",
        "expect_any_of": ["rec-0192", "rec-0158", "rec-0232"],
        "top_k": 14,
        "tier": "required",
    },
]

# Tracked paraphrase gaps — stay in CI reporting even when still failing (not hidden).
RANKING_KNOWN_GAPS: list[dict] = [
    {
        "name": "payment_status_paraphrase",
        "question": "What happens when instalments are debited but payment or SIP status stays pending?",
        "expect_any_of": ["rec-0285", "rec-0310", "rec-0321", "rec-0306"],
        "top_k": 14,
        "tier": "known_gap",
        "notes": "Canonical transaction_refund_vs_status passes; this wording stresses pending/status.",
    },
    {
        "name": "support_mixed_experiences",
        "question": "Do reviews describe Groww support as responsive or hard to reach?",
        "expect_any_of": ["rec-0032", "rec-0047", "rec-0325", "rec-0310"],
        "top_k": 14,
        "tier": "known_gap",
        "notes": "Canonical positive_and_negative_same_topic passes; responsive/reach phrasing is harder.",
    },
]


def _ranking_case_pass(case: dict) -> tuple[bool, list[str], list[str]]:
    hits_list = retrieve(case["question"], lexical_question=case["question"])
    ids = [h["id"] for h in hits_list[: case.get("top_k", 14)]]
    expect = case.get("expect_any_of") or []
    ok = not expect or bool(set(ids).intersection(expect))
    failures: list[str] = []
    if not ok:
        failures.append(
            f"{case['name']}: none of {expect} in top {len(ids)} (got {ids[:4]}…)"
        )
    if case.get("forbid_top1_praise_only") and ids:
        from rerank import PRAISE_DOC, PROBLEM_DOC

        top = hits_list[0]
        text = top.get("text") or ""
        if PRAISE_DOC.search(text) and not PROBLEM_DOC.search(text):
            failures.append(f"{case['name']}: top hit is generic praise ({top['id']})")
            ok = False
    return ok, failures, ids


def run_ranking_checks() -> tuple[list[str], dict[str, float], list[dict]]:
    failures: list[str] = []
    hits = 0
    required = [c for c in RANKING_CASES if c.get("tier", "required") == "required"]
    total = len(required)
    gap_reports: list[dict] = []

    for case in required:
        ok, case_failures, ids = _ranking_case_pass(case)
        failures.extend(case_failures)
        if ok:
            hits += 1

    for case in RANKING_KNOWN_GAPS:
        ok, case_failures, ids = _ranking_case_pass(case)
        expect = case.get("expect_any_of") or []
        ranks = {}
        import retrieval as retrieval_mod

        prev_max = retrieval_mod.MAX_RECORDS
        retrieval_mod.MAX_RECORDS = 200
        try:
            hits_list = retrieve(case["question"], lexical_question=case["question"])
        finally:
            retrieval_mod.MAX_RECORDS = prev_max
        all_ids = [h["id"] for h in hits_list]
        for rid in expect:
            ranks[rid] = all_ids.index(rid) + 1 if rid in all_ids else None
        gap_reports.append(
            {
                "name": case["name"],
                "pass": ok,
                "top_ids": ids[:6],
                "label_ranks": ranks,
                "notes": case.get("notes", ""),
                "failures": case_failures,
            }
        )

    metrics = {"ranking_recall": hits / total if total else 1.0}
    return failures, metrics, gap_reports


def run_evidence_selection_checks() -> tuple[list[str], dict[str, float]]:
    """Verify evidence selection policy (retrieval candidates vs LLM prompt)."""
    from evidence import explain_evidence_selection, select_evidence

    failures: list[str] = []
    retained = 0
    total = 0
    scenarios = [
        (
            "portfolio_worry",
            "What worries users about tracking their portfolio after they start a SIP?",
            ["rec-0192", "rec-0158", "rec-0232"],
        ),
        (
            "transaction_canonical",
            "What do reviews say about money deducted but SIP or payment status not updating?",
            ["rec-0285", "rec-0310", "rec-0306"],
        ),
    ]
    for _name, question, labels in scenarios:
        hits = retrieve(question, lexical_question=question)
        if not hits:
            failures.append(f"evidence {_name}: no retrieval hits")
            continue
        lexical_q = __import__("hybrid_search", fromlist=["expand_query"]).expand_query(
            question
        )
        report = explain_evidence_selection(hits, lexical_q)
        chosen_ids = set(report["prompt_ids"])
        if len(report["prompt_ids"]) > report["max_items"]:
            failures.append(
                f"evidence {_name}: prompt count {len(report['prompt_ids'])} "
                f"exceeds max_items {report['max_items']}"
            )
        top_retrieved = sorted(
            hits,
            key=lambda h: float(h.get("relevance_score", h.get("similarity", 0.0))),
            reverse=True,
        )[:5]
        for hit in top_retrieved:
            total += 1
            if hit["id"] in chosen_ids:
                retained += 1
            elif hit["id"] in report["retrieval_top14"]:
                failures.append(
                    f"evidence {_name}: retrieval rank "
                    f"{report['retrieval_candidates'].index(hit['id']) + 1} "
                    f"{hit['id']} dropped despite top-5 policy"
                )
        for rid in labels:
            rank = next(
                (i + 1 for i, h in enumerate(hits) if h["id"] == rid),
                None,
            )
            if rank and rank <= 12 and rid not in chosen_ids:
                if rid in report["dropped_from_retrieval_top14"]:
                    failures.append(
                        f"evidence {_name}: {rid} at retrieval rank {rank} "
                        f"dropped by max_items={report['max_items']} "
                        f"(prompt has {len(chosen_ids)} ids)"
                    )
    rate = retained / total if total else 1.0
    return failures, {"top5_retained_in_prompt": rate}


def run_evidence_policy_unit_checks() -> list[str]:
    from evidence import explain_evidence_selection

    failures: list[str] = []
    synthetic = [
        {
            "id": f"rec-{i:04d}",
            "text": f"review text {i} sip portfolio",
            "relevance_score": 1.0 - i * 0.05,
            "similarity": 1.0 - i * 0.05,
        }
        for i in range(1, 15)
    ]
    lexical_q = "sip portfolio tracking"
    report = explain_evidence_selection(synthetic, lexical_q, max_items=12)
    if len(report["prompt_ids"]) != 12:
        failures.append(
            f"evidence policy: expected 12 prompt ids, got {len(report['prompt_ids'])}"
        )
    for rid in [f"rec-{i:04d}" for i in range(1, 11)]:
        if rid not in report["prompt_ids"]:
            failures.append(f"evidence policy: {rid} should be retrieval-priority in prompt")
    return failures


def run_citation_validation_checks() -> list[str]:
    from citation_validate import claim_supported_by_source, validate_answer_citations
    from evidence import to_evidence_item

    failures: list[str] = []
    hits = retrieve(
        "What confusion do users report about how much they invested or their portfolio status?",
        lexical_question="What confusion do users report about how much they invested or their portfolio status?",
    )
    if not hits:
        return ["citation checks: no hits"]
    item = to_evidence_item(hits[0])
    words = [w for w in item["text"].split() if len(w) > 4][:8]
    snippet = " ".join(words) if words else "portfolio dashboard invested"
    supported, _ = claim_supported_by_source(
        f"Reviewers report {snippet} [{item['id']}].",
        item["text"],
    )
    if not supported:
        failures.append("citation validator rejected a clearly anchored mock claim")

    decoy = to_evidence_item(hits[-1])
    bad, _ = claim_supported_by_source(
        f"Users praised instant KYC approval [{decoy['id']}].",
        decoy["text"],
    )
    if bad:
        failures.append("citation validator should flag unrelated mock claim")
    issues = validate_answer_citations(
        f"Users praised instant KYC approval [{decoy['id']}].",
        [decoy],
        valid_ids={decoy["id"]},
    )
    if not issues:
        failures.append("citation validator did not flag weak decoy citation")
    return failures


def run_guard_research_checks() -> list[str]:
    failures: list[str] = []
    research_questions = [
        "What do reviewers say they find confusing about returns or portfolio values shown in Groww?",
        "What do commenters in the collected feedback recommend that beginners should do when starting a SIP?",
        "What separates investors who keep a SIP running from those who pause or stop?",
    ]
    for q in research_questions:
        if guard(q):
            failures.append(f"guard blocked corpus research question: {q[:60]}…")
    if not guard("Which fund should I buy for my portfolio?"):
        failures.append("guard should block personal advice question")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="SIP Insight Lab backend evaluation")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Run retrieval and metadata checks only (no Groq answer generation)",
    )
    args = parser.parse_args()

    failures, metrics = run_retrieval_checks()
    rank_failures, rank_metrics, gap_reports = run_ranking_checks()
    failures.extend(rank_failures)
    ev_failures, ev_metrics = run_evidence_selection_checks()
    failures.extend(ev_failures)
    failures.extend(run_motivation_regression_checks())
    failures.extend(run_evidence_policy_unit_checks())
    failures.extend(run_citation_validation_checks())
    failures.extend(run_metadata_checks())
    failures.extend(run_guard_research_checks())

    mode = "retrieval-only" if args.retrieval_only else "full"
    if not args.retrieval_only:
        failures.extend(run_answer_checks())

    recall_pct = metrics["labeled_recall"] * 100.0
    print(
        f"Mode: {mode} | retrieval cases: {len(CASES)} | "
        f"required ranking: {len([c for c in RANKING_CASES if c.get('tier','required')=='required'])} | "
        f"known-gap cases: {len(RANKING_KNOWN_GAPS)} | "
        f"labeled recall: {metrics['labeled_recall']:.2f} | "
        f"required ranking recall: {rank_metrics['ranking_recall']:.2f} | "
        f"top5 retained for prompt: {ev_metrics['top5_retained_in_prompt']:.2f}"
    )
    print("Known-gap ranking (tracked, not omitted):")
    for row in gap_reports:
        status = "PASS" if row["pass"] else "FAIL"
        print(
            f"  [{status}] {row['name']}: top={row['top_ids']} "
            f"label_ranks={row.get('label_ranks', {})} "
            f"— {row.get('notes','')}"
        )
    unresolved = [r for r in gap_reports if not r["pass"]]
    if unresolved:
        print(f"Unresolved known gaps: {len(unresolved)} (see above)")

    if failures:
        print("Evaluation failed:")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)

    print(f"Evaluation passed (labeled recall {recall_pct:.0f}%).")


if __name__ == "__main__":
    main()
