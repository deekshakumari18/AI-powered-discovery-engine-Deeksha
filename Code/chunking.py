"""Phase 2 — Recursive chunking.

Reads Data/raw/loaded.json.
Writes citable chunks to Data/chunks/chunks.json.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOADED_PATH = PROJECT_ROOT / "Data" / "raw" / "loaded.json"
CHUNKS_PATH = PROJECT_ROOT / "Data" / "chunks" / "chunks.json"

# Small enough for a few chunks in a Gemini prompt; large enough for one cited claim.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_keep(text: str, separator: str) -> list[str]:
    if separator == "":
        return list(text)
    if not separator:
        return [text]
    parts = text.split(separator)
    return [p for p in parts if p != ""]


def _join(parts: list[str], separator: str) -> str:
    return separator.join(parts)


def _merge_with_overlap(pieces: list[str], separator: str) -> list[str]:
    if not pieces:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    sep_len = len(separator)

    def current_text() -> str:
        return _join(current, separator)

    def overlap_parts(source: list[str]) -> list[str]:
        if not source or CHUNK_OVERLAP <= 0:
            return []
        kept: list[str] = []
        size = 0
        for part in reversed(source):
            extra = sep_len if kept else 0
            if size + extra + len(part) > CHUNK_OVERLAP and kept:
                break
            kept.append(part)
            size += extra + len(part)
        kept.reverse()
        return kept

    for piece in pieces:
        extra = sep_len if current else 0
        if current and current_len + extra + len(piece) > CHUNK_SIZE:
            chunks.append(current_text())
            current = overlap_parts(current)
            current_len = len(current_text()) if current else 0
            extra = sep_len if current else 0
            while current and current_len + extra + len(piece) > CHUNK_SIZE:
                current = current[1:]
                current_len = len(current_text()) if current else 0
                extra = sep_len if current else 0
        current.append(piece)
        current_len += extra + len(piece)

    if current:
        chunks.append(current_text())
    return chunks


def recursive_split(text: str, separators: list[str] | None = None) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [text]

    seps = list(separators if separators is not None else SEPARATORS)
    if not seps:
        return [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP or CHUNK_SIZE)]

    sep = seps[0]
    rest = seps[1:]
    pieces = _split_keep(text, sep)
    if len(pieces) <= 1:
        return recursive_split(text, rest)

    ready: list[str] = []
    for piece in pieces:
        if len(piece) > CHUNK_SIZE:
            ready.extend(recursive_split(piece, rest))
        else:
            ready.append(piece)
    return _merge_with_overlap(ready, sep)


def chunk_units(units: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for unit in units:
        texts = recursive_split(unit["text"])
        if not texts:
            continue
        embed_source = unit.get("embed_text") or unit["text"]
        embed_chunks = (
            recursive_split(embed_source) if embed_source != unit["text"] else texts
        )
        for i, text in enumerate(texts, start=1):
            embed_text = embed_chunks[i - 1] if i - 1 < len(embed_chunks) else embed_source
            chunks.append(
                {
                    "id": f"{unit['id']}-c{i:02d}",
                    "text": text,
                    "embed_text": embed_text,
                    "source_name": unit["source_name"],
                    "section": unit.get("section"),
                    "source_type": unit.get("source_type"),
                    "english_translation": unit.get("english_translation"),
                    "citation_url": unit.get("citation_url"),
                    "review_date": unit.get("review_date"),
                    "last_updated": unit["last_updated"],
                    "parent_id": unit["id"],
                }
            )
    return chunks


def load_units(path: Path | None = None) -> list[dict]:
    src = path or LOADED_PATH
    if not src.is_file():
        raise FileNotFoundError(f"Run loading first. Missing {src}")
    payload = json.loads(src.read_text(encoding="utf-8"))
    return payload["units"]


def save_chunks(chunks: list[dict], dest: Path | None = None) -> Path:
    out = dest or CHUNKS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "separators": SEPARATORS,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    units = load_units()
    chunks = chunk_units(units)
    path = save_chunks(chunks)
    cited = sum(1 for c in chunks if c.get("citation_url") or c.get("source_name"))
    split_units = sum(1 for u in units if len(u["text"]) > CHUNK_SIZE)
    print(f"Chunked {len(units)} units into {len(chunks)} chunks ({split_units} units split)")
    print(f"Citable chunks: {cited}/{len(chunks)}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
