"""Phase 3 — Embedding.

Reads Data/chunks/chunks.json.
Writes one vector per chunk to Data/embeddings/embeddings.json.
Model: sentence-transformers/all-MiniLM-L6-v2 (same model for queries later).
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "Data" / "chunks" / "chunks.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "Data" / "embeddings" / "embeddings.json"

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_ID)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return [row.tolist() for row in vectors]


def embed_query(question: str) -> list[float]:
    return embed_texts([question])[0]


def load_chunks(path: Path | None = None) -> list[dict]:
    src = path or CHUNKS_PATH
    if not src.is_file():
        raise FileNotFoundError(f"Run chunking first. Missing {src}")
    payload = json.loads(src.read_text(encoding="utf-8"))
    return payload["chunks"]


def embed_chunks(chunks: list[dict]) -> dict:
    texts = [chunk.get("embed_text") or chunk["text"] for chunk in chunks]
    vectors = embed_texts(texts)
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding count does not match chunk count")
    dims = {len(v) for v in vectors}
    if len(dims) != 1:
        raise RuntimeError(f"Mixed embedding sizes: {dims}")
    rows = []
    for chunk, vector in zip(chunks, vectors):
        rows.append(
            {
                "id": chunk["id"],
                "embedding": vector,
            }
        )
    return {
        "model_id": MODEL_ID,
        "embedding_dim": dims.pop(),
        "chunk_count": len(rows),
        "rows": rows,
    }


def save_embeddings(payload: dict, dest: Path | None = None) -> Path:
    out = dest or EMBEDDINGS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def main() -> None:
    chunks = load_chunks()
    payload = embed_chunks(chunks)
    path = save_embeddings(payload)
    print(
        f"Embedded {payload['chunk_count']} chunks "
        f"with {payload['model_id']} (dim={payload['embedding_dim']})"
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
