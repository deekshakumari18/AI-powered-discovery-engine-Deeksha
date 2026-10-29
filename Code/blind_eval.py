"""Blind evaluation diagnostics — tests only, not production routing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_search import contextualize_question, expand_query
from retrieval import guard, retrieve, MIN_SIMILARITY
from evidence import format_citations_for_ids, to_evidence_item, select_evidence
from hybrid_search import expand_query

# Unseen questions (not duplicated in tests.py CASES wording where possible).
BLIND: list[dict] = [
    {
        "id": "b01",
        "question": "What separates investors who keep a SIP running from those who pause or stop, and does that differ by age or by who influenced them to start?",
        "expect_any_of": ["rec-0221", "rec-0243", "rec-0004"],
    },
    {
        "id": "b02",
        "question": "After the first few instalments, what do investors check inside the app, how often, and what do they do outside the app when they feel unsure?",
        "expect_any_of": ["rec-0242", "rec-0084", "rec-0222"],
    },
    {
        "id": "b03",
        "question": "Are SIP stops mostly caused by Autopay or mandate failures, or by investors choosing to pause when markets fall? Which of those matters more for staying invested 12 months?",
        "expect_any_of": ["rec-0321", "rec-0285", "rec-0003"],
    },
    {
        "id": "b04",
        "question": "Which recurring unmet needs show up when people talk about not continuing a SIP for a full year?",
        "expect_any_of": ["rec-0321", "rec-0269", "rec-0221"],
    },
    {
        "id": "b05",
        "question": "What do investors say goes wrong with SIP AutoPay, and how is that different from complaints about customer support?",
        "expect_any_of": ["rec-0321", "rec-0032", "rec-0310"],
    },
    {
        "id": "b06",
        "question": "What problems do beginners describe when starting on the app, compared with complaints about wrong balances on the dashboard?",
        "expect_any_of": ["rec-0032", "rec-0158", "rec-0192"],
    },
    {
        "id": "b07",
        "question": "What do reviewers say they find confusing about returns or portfolio values shown in Groww?",
        "expect_any_of": ["rec-0158", "rec-0192", "rec-0221"],
        "must_not_guard": True,
    },
    {
        "id": "b08",
        "question": "What do commenters in the collected feedback recommend that beginners should do when starting a SIP?",
        "expect_any_of": ["rec-0020", "rec-0016", "rec-0014"],
        "must_not_guard": True,
    },
    {
        "id": "b09",
        "question": "Based on the feedback, what do people report across onboarding, KYC, completing a first SIP, and getting help from support?",
        "expect_any_of": ["rec-0032", "rec-0024", "rec-0047"],
    },
    {
        "id": "b10",
        "question": "What do people report about failed KYC or verification steps when trying to invest in mutual funds?",
        "expect_any_of": ["rec-0032", "rec-0024"],
    },
    {
        "id": "b11",
        "question": "When markets fall, what do people in the corpus say about pausing or continuing their SIP?",
        "expect_any_of": ["rec-0003", "rec-0001", "rec-0221"],
    },
    {
        "id": "b12",
        "question": "What do Hindi or Hinglish YouTube comments say about SIP investing when an English translation is available?",
        "expect_any_of": [],  # any youtube hit with translation optional
        "min_hits": 1,
    },
    {
        "id": "b13",
        "question": "What about when markets drop?",
        "history": [{"role": "user", "content": "What leads people to pause or stop a SIP?"}],
        "expect_any_of": ["rec-0003", "rec-0221", "rec-0321"],
    },
    {
        "id": "b14",
        "question": "What worries users about tracking their portfolio after they start a SIP?",
        "expect_any_of": ["rec-0158", "rec-0192", "rec-0043"],
    },
    {
        "id": "b15",
        "question": "What happens when customer support does not fix a failed SIP deduction or autopay problem?",
        "expect_any_of": ["rec-0321", "rec-0285", "rec-0325"],
    },
]


def diagnose_case(case: dict) -> dict:
    q = case["question"]
    history = case.get("history")
    blocked = guard(q)
    guard_fail = bool(case.get("must_not_guard") and blocked)
    search_q = contextualize_question(q, history) if history else q
    hits = retrieve(search_q, lexical_question=q) if not blocked else []
    ids = [h["id"] for h in hits]
    links = ""
    if hits:
        lexical_q = expand_query(q)
        ev = select_evidence(hits, lexical_q)
        items = [to_evidence_item(h) for h in ev]
        links = format_citations_for_ids(items, [h["id"] for h in ev[:3]])
    expect = case.get("expect_any_of") or []
    recall = bool(set(ids) & set(expect)) if expect else None
    if case.get("min_hits") and len(hits) < case["min_hits"]:
        recall = False
    return {
        "id": case["id"],
        "blocked": blocked,
        "guard_fail": guard_fail,
        "hit_count": len(hits),
        "top_sim": hits[0]["similarity"] if hits else 0.0,
        "top_ids": ids[:8],
        "expect_recall": recall,
        "source_links_preview": links[:200],
        "unique_urls": len({h.get("citation_url") for h in hits if h.get("citation_url")}),
    }


def main() -> None:
    rows = [diagnose_case(c) for c in BLIND]
    out = PROJECT_ROOT / "Data" / "eval" / "blind_diagnostic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    guards = [r for r in rows if r["blocked"]]
    guard_fail = [r for r in rows if r.get("guard_fail")]
    empty = [r for r in rows if not r["blocked"] and r["hit_count"] == 0]
    miss = [r for r in rows if r.get("expect_recall") is False]
    print(
        f"cases={len(rows)} guard_blocked={len(guards)} guard_fail={len(guard_fail)} "
        f"no_hits={len(empty)} expect_miss={len(miss)}"
    )
    for r in rows:
        flag = []
        if r.get("guard_fail"):
            flag.append("GUARD_FAIL")
        elif r["blocked"]:
            flag.append("GUARD")
        if r["hit_count"] == 0 and not r["blocked"]:
            flag.append("NO_HITS")
        if r.get("expect_recall") is False:
            flag.append("MISS")
        print(f"{r['id']} {' '.join(flag) or 'ok'} hits={r['hit_count']} top={r['top_ids'][:3]}")


if __name__ == "__main__":
    main()
