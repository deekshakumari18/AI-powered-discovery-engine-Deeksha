# PRD: AI-Powered Discovery Engine

**Product:** Facts-only RAG chatbot for SIP continuation insights  
**Status:** Draft (project MVP)  
**Source:** `Docs/problemstatement.txt`  
**Corpus:** `Groww_SIP_Investor_Journey_Review_Corpus.docx`

**North-star metric:** Increase the % of first-time mutual fund investors who still have an **active SIP** 12 months after starting. Active SIP = at least one successful instalment in the last 30 days.

---

## 1. Problem

Starting a SIP is easy. Staying with it for 12 months is not. First-time investors pause, reduce, or lapse when statements look underwhelming, life gets busy, or the app feels confusing.

The review corpus holds evidence about why people start, what they expect, and why they drop off — but it is hard to query as evidence. We need a simple way to **identify, quantify where possible, and compare opportunity areas** that could move 12-month SIP continuation — without turning the tool into investment advice.

---

## 2. Solution

A **simple Streamlit chatbot (RAG)** that answers only from the uploaded corpus.

It is not a generic review summariser or sentiment tool. It should help surface **why people start, what they expect, what they do after starting, why they pause/stop, and which unmet needs show up consistently** — so we can compare opportunity areas against the 12-month continuation goal.

---

## 3. Users

| Who | Need |
| --- | --- |
| Product / research (project user) | Ask discovery questions and get short, cited, facts-only answers from the corpus |

No login. No personal accounts. No PII collection.

---

## 4. Goals (in scope)

1. Answer **only from the provided document(s)**.
2. Show **one clear citation link** in every answer.
3. Keep answers **≤ 3 sentences**, plus `Last updated from sources: <date>`.
4. Support discovery questions (if the corpus supports them), including:
   - Why people start a SIP and what they expect
   - Behaviour change in the months after starting
   - What leads investors to pause, reduce, or stop
   - What they check, and how often
   - In-app confusion / worry
   - What they do outside the app when unsure
   - Continuers vs drop-offs
   - Segment differences (income stability, age, source of money, who influenced them)
   - Consistent unmet needs
5. Stay lightweight: Hugging Face embeddings + ChromaDB + Gemini + Streamlit.

---

## 5. Non-goals

- Deep analytics dashboards, A/B experiments, or production infra
- Computing or comparing fund **returns / performance**
- Opinion, portfolio construction, or investment advice
- App back-end screenshots or third-party blogs as sources
- Auth, multi-user, or complex UI

---

## 6. Constraints

| Constraint | Rule |
| --- | --- |
| Public sources only | Use the provided corpus. No app back-end screenshots. No third-party blogs. |
| No PII | Do not accept or store PAN, Aadhaar, account numbers, OTPs, emails, or phone numbers. |
| No performance claims | Do not compute/compare returns. If asked, link to an official factsheet. |
| Clarity | Answers ≤ 3 sentences. Include `Last updated from sources: `. |
| Facts only | No invented facts. If the corpus does not cover it, say so. |

---

## 7. Product behaviour

### UI (tiny)

- Welcome line
- **3 example questions**
- Note: **“Facts only. No investment advice.”**
- Chat input + answer area
- Simple Streamlit layout; calm finance-trust colours (e.g. navy + off-white)

**Example questions:**

1. Why do first-time investors start a SIP, and what do they expect?  
2. What leads people to pause, reduce, or stop a SIP?  
3. What confuses or worries users inside investing apps after they start?

### Answer rules

| Rule | Behaviour |
| --- | --- |
| Grounding | Use retrieved chunks only |
| Length | ≤ 3 sentences |
| Citation | Exactly **one** clear citation per answer |
| Freshness | Include `Last updated from sources: ` |
| Performance questions | Do not invent numbers; point to official factsheet |
| Opinion / portfolio | Refuse politely with a facts-only message + one educational link |
| Missing evidence | Say the documents do not cover it; do not guess |

### Discovery value (beyond sentiment)

Where the corpus allows, answers should:

- Name the **opportunity** (e.g. statement anxiety in months 1–3)
- **Quantify** if counts/frequency appear in the data
- **Compare** opportunity areas by likely effect on 12-month continuation
- Call out **gaps** (including segments) only if the data supports them

---

## 8. Technical approach (project-level)

| Step | Choice |
| --- | --- |
| Ingest | Upload the corpus document |
| Chunk | Recursive chunking (split on headings/paragraphs, then sentences, then words; small overlap) |
| Embed | `sentence-transformers/all-MiniLM-L6-v2` |
| Store | ChromaDB |
| Generate | Google Gemini (API key) |
| App | Streamlit |

Flow: question → embed → retrieve chunks → prompt Gemini with chunks + rules → short cited answer.

---

## 9. Success (MVP done when)

- [ ] Corpus can be ingested and queried in Streamlit
- [ ] Answers stay on-corpus, ≤ 3 sentences, with one citation and last-updated line
- [ ] Opinion / portfolio questions are refused with an educational link
- [ ] Returns questions do not invent numbers; they defer to an official factsheet
- [ ] The 3 example questions return useful, cited answers
- [ ] UI is tiny: welcome + 3 examples + “Facts only. No investment advice.”
- [ ] No PII is requested or stored

---

## 10. Open points (decide in build, do not expand scope)

- Citation format for a local Word corpus (file + section vs source URL inside the doc)
- Exact `Last updated from sources` date (ingest date vs date in the corpus)
- Educational link on refusal (e.g. SEBI / AMFI investor education)
- Official factsheet URL when returns are asked
