"""Classify user intent for guards — corpus research vs personal advice/calculation."""

from __future__ import annotations

import re

# User is asking what the collected feedback says (not asking the bot to decide for them).
CORPUS_RESEARCH = re.compile(
    r"\b("
    r"what do (?:the )?(?:reviews|reviewers|people|users|commenters|investors)|"
    r"what (?:does|do) (?:the )?(?:corpus|feedback|comments|discussions|collected feedback)|"
    r"what (?:are|is) .{0,50}(?:reported|described|mentioned|say about)|"
    r"what (?:problems|issues|confusion|worries|complaints) .{0,40}(?:report|describe|say)|"
    r"what do reviewers say|what do commenters|what happens when|"
    r"based on (?:the )?(?:feedback|reviews|corpus|documents)|"
    r"in the (?:collected )?(?:feedback|corpus|reviews)|"
    r"when markets (?:fall|drop)|"
    r"walk through|across onboarding"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)

FOLLOW_UP_RESEARCH = re.compile(
    r"\b(what about|how about|and what|when markets|same for|follow[- ]?up)\b",
    re.IGNORECASE,
)

PERSONAL_ADVICE = re.compile(
    r"\b("
    r"should i|should we|what should i|what should we|"
    r"which fund should i|which sip should i|which fund|which sip|best fund|"
    r"my portfolio|portfolio review|asset allocation|"
    r"where should i invest|what should i buy|investment advice for me"
    r")\b",
    re.IGNORECASE,
)

PERSONAL_RETURNS = re.compile(
    r"\b("
    r"calculate|compute|compare my|what is my return|what was my return|"
    r"what(?:'s| is) the (?:cagr|xirr)|show me (?:the )?nav|factsheet for my"
    r")\b",
    re.IGNORECASE,
)

PERFORMANCE_METRIC = re.compile(
    r"\b(cagr|xirr|factsheet)\b",
    re.IGNORECASE,
)


def is_corpus_research(question: str, *, has_history: bool = False) -> bool:
    text = question.strip()
    if not text:
        return False
    if CORPUS_RESEARCH.search(text):
        return True
    if has_history and FOLLOW_UP_RESEARCH.search(text):
        return True
    if re.search(
        r"\bwhy do (?:first[- ]time |new )?(?:investors|people|users|folks)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\bwhy (?:do )?(?:people|users|investors|folks)\b.{0,40}\b(?:start|begin)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\bwhat (?:leads|separates|causes|worries|confuses|happens)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    if re.search(r"\b(?:after|during|when).{0,60}\b(?:sip|instalment|invest)\b", text, re.I):
        return True
    return False


def should_block_returns(question: str) -> bool:
    if is_corpus_research(question):
        if PERSONAL_RETURNS.search(question):
            return True
        if PERFORMANCE_METRIC.search(question):
            return True
        return False
    if PERSONAL_RETURNS.search(question):
        return True
    if re.search(r"\breturns?\b", question, re.IGNORECASE):
        if re.search(r"\b(?:compare|calculate|compute|performance of)\b", question, re.I):
            return True
        if re.search(r"\b(?:my|your)\b.{0,20}\breturns?\b", question, re.I):
            return True
    if PERFORMANCE_METRIC.search(question):
        return True
    return False


def should_block_advice(question: str) -> bool:
    if PERSONAL_ADVICE.search(question):
        return True
    if is_corpus_research(question):
        return False
    if re.search(r"\brecommend\b", question, re.IGNORECASE):
        if re.search(r"\bwhat do\b", question, re.IGNORECASE):
            return False
        return True
    return False
