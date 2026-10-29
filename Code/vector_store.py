"""Phase 4 — Vector store.

Reads embeddings + chunk metadata.
Writes one local ChromaDB collection to Data/vectordb.
Rebuilds the collection on each run so unknown files are not mixed in.
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "Data" / "chunks" / "chunks.json"
THEMES_PATH = PROJECT_ROOT / "Data" / "chunks" / "themes.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "Data" / "embeddings" / "embeddings.json"
VECTORDB_DIR = PROJECT_ROOT / "Data" / "vectordb"
COLLECTION_NAME = "groww_sip_corpus"


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _meta_value(value) -> str:
    if value is None:
        return ""
    return str(value)


def _themes_by_record() -> dict[str, dict]:
    if not THEMES_PATH.is_file():
        return {}
    payload = _load_json(THEMES_PATH)
    labels = {item["theme_id"]: item for item in payload.get("themes", [])}
    mapped = {}
    for record_id, theme_id in payload.get("by_record", {}).items():
        theme = labels.get(theme_id, {})
        mapped[record_id] = {
            "theme_id": theme_id,
            "theme": theme.get("label", ""),
            "theme_size": theme.get("size", 0),
        }
    return mapped


def build_records() -> tuple[list[str], list[list[float]], list[str], list[dict], str]:
    chunks = {c["id"]: c for c in _load_json(CHUNKS_PATH)["chunks"]}
    themes = _themes_by_record()
    payload = _load_json(EMBEDDINGS_PATH)
    model_id = payload["model_id"]
    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for row in payload["rows"]:
        chunk = chunks.get(row["id"])
        if chunk is None:
            raise KeyError(f"Embedding id {row['id']} has no chunk metadata")
        ids.append(chunk["id"])
        embeddings.append(row["embedding"])
        documents.append(chunk["text"])
        theme = themes.get(chunk.get("parent_id"), {})
        metadatas.append(
            {
                "parent_id": _meta_value(chunk.get("parent_id")),
                "source_type": _meta_value(chunk.get("source_type")),
                "source_name": _meta_value(chunk.get("source_name")),
                "section": _meta_value(chunk.get("section")),
                "last_updated": _meta_value(chunk.get("last_updated")),
                "citation_url": _meta_value(chunk.get("citation_url")),
                "theme_id": _meta_value(theme.get("theme_id")),
                "theme": _meta_value(theme.get("theme")),
                "theme_size": int(theme.get("theme_size") or 0),
            }
        )
    return ids, embeddings, documents, metadatas, model_id


def get_client() -> chromadb.PersistentClient:
    VECTORDB_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(VECTORDB_DIR))


def rebuild_collection() -> chromadb.Collection:
    ids, embeddings, documents, metadatas, model_id = build_records()
    client = get_client()
    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"model_id": model_id, "corpus": "Groww_SIP_Investor_Journey_Review_Corpus.docx"},
    )
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return collection


def smoke_query(collection: chromadb.Collection | None = None) -> dict:
    client = get_client()
    coll = collection or client.get_collection(COLLECTION_NAME)
    probe_id = coll.peek(limit=1)["ids"][0]
    got = coll.get(ids=[probe_id], include=["documents", "metadatas", "embeddings"])
    text = got["documents"][0] if got["documents"] else None
    if not text:
        raise RuntimeError("Smoke query returned an id with no chunk text")
    return {
        "id": probe_id,
        "text": text,
        "metadata": got["metadatas"][0],
        "count": coll.count(),
    }


def main() -> None:
    collection = rebuild_collection()
    result = smoke_query(collection)
    print(f"Collection {COLLECTION_NAME}: {result['count']} items in {VECTORDB_DIR}")
    print(f"Smoke id: {result['id']}")
    print(f"Smoke text (first 160 chars): {result['text'][:160]!r}")
    print(f"Smoke citation: {result['metadata'].get('citation_url')}")
    print(f"Last updated: {result['metadata'].get('last_updated')}")


if __name__ == "__main__":
    main()
