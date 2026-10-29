"""BM25 index over source records (unit-level embed_text)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from hybrid_search import tokenize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOADED_PATH = PROJECT_ROOT / "Data" / "raw" / "loaded.json"

_index: "BM25Index | None" = None


class BM25Index:
    def __init__(
        self,
        ids: list[str],
        texts: list[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.ids = ids
        self.k1 = k1
        self.b = b
        self.doc_len: list[int] = []
        self.tf: list[dict[str, int]] = []
        self.doc_freq: dict[str, int] = {}
        for text in texts:
            tokens = tokenize(text)
            self.doc_len.append(len(tokens) or 1)
            freq: dict[str, int] = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
            self.tf.append(freq)
        self.n_docs = len(texts)
        self.avgdl = sum(self.doc_len) / self.n_docs if self.n_docs else 1.0
        for freq in self.tf:
            for token in freq:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    def _idf(self, token: str) -> float:
        df = self.doc_freq.get(token, 0)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def score_query(self, query: str) -> list[tuple[str, float]]:
        qtokens = tokenize(query)
        if not qtokens:
            return []
        raw: list[tuple[str, float]] = []
        for i, rid in enumerate(self.ids):
            total = 0.0
            dl = self.doc_len[i]
            for token in qtokens:
                tf = self.tf[i].get(token, 0)
                if tf == 0:
                    continue
                idf = self._idf(token)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                total += idf * tf * (self.k1 + 1) / denom
            raw.append((rid, total))
        raw.sort(key=lambda row: row[1], reverse=True)
        return raw

    def normalized_scores(self, query: str, top_k: int = 120) -> dict[str, float]:
        ranked = self.score_query(query)
        if not ranked:
            return {}
        top = ranked[0][1]
        if top <= 0:
            return {}
        out: dict[str, float] = {}
        for rid, score in ranked:
            norm = min(1.0, score / top)
            if len(out) < top_k or norm >= 0.22:
                out[rid] = norm
        return out

    def normalized_score_for(self, query: str, record_id: str) -> float:
        ranked = self.score_query(query)
        if not ranked or ranked[0][1] <= 0:
            return 0.0
        top = ranked[0][1]
        for rid, score in ranked:
            if rid == record_id:
                return min(1.0, score / top)
        return 0.0


def get_bm25_index() -> BM25Index:
    global _index
    if _index is not None:
        return _index
    payload = json.loads(LOADED_PATH.read_text(encoding="utf-8"))
    ids: list[str] = []
    texts: list[str] = []
    for unit in payload.get("units", []):
        ids.append(unit["id"])
        texts.append(unit.get("embed_text") or unit.get("text") or "")
    _index = BM25Index(ids, texts)
    return _index
