# Architecture: AI-Powered Discovery Engine

**Status:** Draft (project MVP)  
**Bound to:** `Docs/PRD.md` only  
**Out of bound:** anything in PRD §5 Non-goals (dashboards, auth, multi-user, production infra, returns math, third-party blogs)

This is a single-user Streamlit RAG app. One corpus in, short cited answers out.

---

## 1. System snapshot

```
[Streamlit UI]
     |  question (no PII)
     v
[Retrieval] --embed query--> [ChromaDB]
     |  top chunks + metadata
     v
[Gemini prompt + PRD answer rules]
     |
     v
[Answer ≤3 sentences + 1 citation + Last updated from sources]
```

**Stack (PRD §8):** document upload → recursive chunking → `sentence-transformers/all-MiniLM-L6-v2` → ChromaDB → Gemini → Streamlit.

**Fixed product rules on every generated answer (PRD §6–7):** grounded on retrieved chunks only; ≤ 3 sentences; exactly one citation; `Last updated from sources:`; refuse opinion/portfolio with an educational link; no return numbers (defer to official factsheet); if chunks do not cover it, say so.

---

## 2. Components (MVP only)

| Component | Role |
| --- | --- |
| Corpus file | `Groww_SIP_Investor_Journey_Review_Corpus.docx` |
| Loader | Read the uploaded Word file into text + source metadata |
| Chunker | Recursive chunking: split into retrieval-sized pieces that still carry a citation handle |
| Embedder | `all-MiniLM-L6-v2` for documents and queries (same model) |
| Vector store | Local ChromaDB collection of chunk embeddings + metadata |
| Retriever | Embed the question, return the most similar chunks |
| Generator | Gemini, prompted with chunks + PRD answer rules |
| UI | Tiny Streamlit: welcome, 3 example questions, facts-only note, chat |

No login, no user store, no PII fields. Gemini API key is environment config, not UI input of personal data.

---

## 3. Phase 1 — Data loading

**Purpose:** Get the PRD corpus into memory once, with enough metadata to cite later.

**In:** `Groww_SIP_Investor_Journey_Review_Corpus.docx` (upload / local path).  
**Out:** Plain text units + metadata: `source_name`, optional heading/section, `last_updated` (ingest date or date found in the file — PRD open point).

**Rules from PRD:**

- Load only this provided document. Do not scrape blogs or app back-end screenshots.
- Do not parse or persist PAN, Aadhaar, account numbers, OTPs, emails, or phone numbers. If such strings appear, drop or redact them at load time; do not write them to ChromaDB.
- Keep original wording. Loading is not summarisation or sentiment.

**Done when:** The docx text is readable end-to-end and each unit has a citation handle (filename and/or section; URL if the corpus itself contains one).

---

## 4. Phase 2 — Chunking

**Purpose:** Cut the loaded text into pieces that retrieval can match to discovery questions, without losing the citation.

**In:** Loaded text + metadata.  
**Out:** Chunks `{id, text, source_name, section?, last_updated}`.

**Approach (PRD: recursive chunking):**

- Use recursive character splitting: try separators in order (`\n\n`, `\n`, `. `, space, then characters) until each piece is under the chunk size.
- Keep a small overlap so a sentence is not lost across two retrieval misses.
- Prefer heading / review boundaries when they appear as paragraph breaks in the Word file.
- Chunk size should stay small enough that a few retrieved chunks fit in a Gemini prompt, and large enough that a chunk can support one cited claim.
- Every chunk keeps **one** citation target (the later answer will pick exactly one link).

**Not in this phase:** embeddings, LLM calls, sentiment scores, opportunity scoring.

**Done when:** The full corpus is represented as chunks; no orphan text; every chunk is citable.

---

## 5. Phase 3 — Embedding

**Purpose:** Turn chunk text into vectors with the PRD embedding model.

**In:** Chunk texts.  
**Out:** One vector per chunk, same dimensionality for all rows.

**Choice (PRD):** `sentence-transformers/all-MiniLM-L6-v2`.

**Rules:**

- Embed **chunk text only**, not generated summaries.
- Use the **same model** later for the user question (PRD flow: question → embed → retrieve).
- Run locally / lightweight; no second embedding vendor.

**Done when:** Every chunk has an embedding and the model id is recorded so query-time embeddings cannot silently drift.

---

## 6. Phase 4 — Vector store

**Purpose:** Persist chunk vectors so Streamlit can query them without re-embedding the whole file every question.

**In:** Embeddings + chunk metadata.  
**Out:** One ChromaDB collection.

**Choice (PRD):** ChromaDB, local.

**Store per item:** `id`, embedding, `text`, `source_name`, `section` (if any), `last_updated`.

**Rules:**

- Single collection for this corpus (project MVP).
- Rebuild the collection when the docx is re-uploaded; do not mix unknown files.
- Metadata must be enough to render **one citation** and `Last updated from sources:`.
- No PII in stored metadata.

**Done when:** A smoke query against the collection returns the stored chunk text, not just an id.

---

## 7. Phase 5 — Retrieval logic

**Purpose:** For one user question, fetch only corpus evidence, then generate a PRD-compliant answer (or a PRD-compliant refusal).

**In:** User question from Streamlit.  
**Out:** Short answer string for the UI.

**Flow (PRD §8):**

1. **Guard the question (before retrieve).**
   - If the input looks like PII (PAN, Aadhaar, account number, OTP, email, phone): do not store it; do not retrieve; tell the user we do not accept personal data.
   - If the question is opinion / portfolio / investment advice: do not retrieve for advice; refuse politely, facts-only, with **one educational link**.
   - If the question asks to compute or compare **returns / performance**: do not retrieve for numbers; do not calculate; point to the **official factsheet** link (PRD open point for the URL).
2. **Embed the question** with `all-MiniLM-L6-v2`.
3. **Query ChromaDB** for the top-k similar chunks (small k; enough evidence, not the whole file).
4. **Grounding check.** If similarity is weak or chunks are off-topic: do not guess. Return that the documents do not cover it.
5. **Prompt Gemini** with: retrieved chunk texts, the question, and hard rules:
   - use retrieved chunks only
   - ≤ 3 sentences
   - name / quantify / compare **opportunity areas vs 12-month SIP continuation** only when the chunks support it
   - do not invent counts, segments, or causes
   - exactly **one** citation (from the best supporting chunk’s metadata)
   - append `Last updated from sources: <date>`
6. **Return** that string to Streamlit. UI does not add extra analysis.

**UI contract (not a separate product):** welcome line, three example questions from PRD §7, note “Facts only. No investment advice.”, chat input, answer area.

**Done when:** A discovery question returns a grounded, cited, ≤3-sentence answer; a portfolio question is refused; a returns question defers to the factsheet.

---

## 8. Phase 6 — Retrieval testing

**Purpose:** Prove retrieval + rules against the PRD success list. This is a project checklist, not a production eval platform.

**Must pass (PRD §9):**

| Test | Expectation |
| --- | --- |
| Ingest | Corpus is loaded, chunked, embedded, and queryable from Streamlit |
| Example Q1 | “Why do first-time investors start a SIP, and what do they expect?” → useful, on-corpus, cited |
| Example Q2 | “What leads people to pause, reduce, or stop a SIP?” → useful, on-corpus, cited |
| Example Q3 | “What confuses or worries users inside investing apps after they start?” → useful, on-corpus, cited |
| Format | ≤ 3 sentences, exactly one citation, `Last updated from sources:` present |
| Grounding | Answer claims appear in retrieved chunks; no extra facts |
| Missing evidence | Off-corpus question → “documents do not cover it”, no guess |
| Opinion / portfolio | Polite refusal + educational link; no advice |
| Returns | No invented numbers; official factsheet pointed to |
| PII | UI does not request PII; PII-like input is not stored |
| UI | Welcome + 3 examples + “Facts only. No investment advice.” |

**How to test (keep it simple):**

- Run the three example questions and save retrieved chunk ids + final answer.
- Spot-check that the citation matches the chunk used.
- Run one refusal case (portfolio) and one returns case.
- Run one question the corpus cannot answer.
- Confirm no PAN/Aadhaar/account/OTP/email/phone fields exist in the UI or Chroma metadata.

**Done when:** The PRD MVP checklist is ticked. Do not add extra metrics, dashboards, or A/B tests.

---

## 9. Explicitly not in this architecture

Per PRD non-goals and open points: no extra data sources, no performance engines, no multi-tenant design, no auth, no analytics layer. Citation URL shape, last-updated date source, educational link, and factsheet URL are **build-time constants**, not new services.

---

## 10. Repo layout

```
Docs/
Code/
  loading.py
  chunking.py
  embedding.py
  vector_store.py
  retrieval.py
  ui.py
  tests.py
Data/
  raw/
  chunks/
  embeddings/
  vectordb/
```

Put the corpus in `Data/raw/`. Rebuild `Data/vectordb/` when the docx is re-uploaded.
