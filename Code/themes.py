"""Theme grouping from review meaning.

Ignores topic labels already in the corpus. Embeds each review body,
clusters those vectors, and names each cluster from the comments themselves.
Writes Data/chunks/themes.json, then rebuilds Chroma with a theme on every row.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedding import MODEL_ID, get_model
from vector_store import rebuild_collection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

LOADED_PATH = PROJECT_ROOT / "Data" / "raw" / "loaded.json"
THEMES_PATH = PROJECT_ROOT / "Data" / "chunks" / "themes.json"
ORIGINAL = re.compile(r"Original text:\s*(.*?)(?:\nSource link:|$)", re.DOTALL)


def review_body(text: str) -> str:
    match = ORIGINAL.search(text)
    body = match.group(1).strip() if match else text.strip()
    return re.sub(r"\s+", " ", body)


def load_reviews() -> list[dict]:
    payload = json.loads(LOADED_PATH.read_text(encoding="utf-8"))
    reviews = []
    for unit in payload["units"]:
        body = review_body(unit["text"])
        if body:
            reviews.append({"id": unit["id"], "text": body})
    return reviews


def _choose_k(vectors: np.ndarray) -> int:
    best_k, best_score = 8, -1.0
    for k in range(6, 11):
        labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(vectors)
        score = float(silhouette_score(vectors, labels))
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def cluster_reviews(reviews: list[dict]) -> tuple[np.ndarray, np.ndarray, int]:
    model = get_model()
    vectors = model.encode(
        [row["text"] for row in reviews],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    k = _choose_k(vectors)
    model_km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = model_km.fit_predict(vectors)
    return vectors, labels, k


def _examples(reviews: list[dict], vectors: np.ndarray, labels: np.ndarray, cluster: int) -> list[str]:
    idxs = np.where(labels == cluster)[0]
    centroid = vectors[idxs].mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
    scores = vectors[idxs] @ centroid
    order = idxs[np.argsort(scores)[::-1][:4]]
    return [reviews[i]["text"][:400] for i in order]


def _name_themes(groups: list[dict]) -> dict[str, str]:
    from groq_client import generate

    blocks = []
    for group in groups:
        samples = "\n".join(f"- {text}" for text in group["examples"])
        blocks.append(f"Group {group['theme_id']} ({group['size']} comments):\n{samples}")
    prompt = (
        "These groups of investor comments were formed only because the comments "
        "are similar in meaning. Ignore any topic headings that may have existed "
        "in a source file.\n"
        "Name each group in 3 to 6 words from what the comments mean.\n"
        "Labels must differ from each other. Do not give advice or mention returns.\n"
        "Reply with JSON only: "
        '[{"id": "t01", "label": "short theme name"}]\n\n'
        + "\n\n".join(blocks)
    )
    raw = generate(prompt)
    start, end = raw.find("["), raw.rfind("]")
    parsed = json.loads(raw[start : end + 1])
    return {item["id"]: str(item["label"]).strip() for item in parsed}


def build_themes() -> dict:
    reviews = load_reviews()
    vectors, labels, k = cluster_reviews(reviews)
    groups = []
    for cluster in range(k):
        idxs = np.where(labels == cluster)[0]
        centroid = vectors[idxs].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        groups.append(
            {
                "theme_id": f"t{cluster + 1:02d}",
                "size": int(len(idxs)),
                "centroid": centroid.tolist(),
                "examples": _examples(reviews, vectors, labels, cluster),
            }
        )
    names = _name_themes(groups)
    by_record = {}
    for review, label in zip(reviews, labels):
        by_record[review["id"]] = f"t{int(label) + 1:02d}"
    for group in groups:
        group["label"] = names.get(group["theme_id"], group["theme_id"])
    return {
        "model_id": MODEL_ID,
        "k": k,
        "method": "kmeans on review-text embeddings; labels from comment meaning",
        "themes": groups,
        "by_record": by_record,
    }


def save_themes(payload: dict) -> Path:
    THEMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    THEMES_PATH.write_text(json.dumps(payload), encoding="utf-8")
    return THEMES_PATH


def main() -> None:
    payload = build_themes()
    path = save_themes(payload)
    rebuild_collection()
    print(f"Grouped {len(payload['by_record'])} reviews into {payload['k']} themes")
    for theme in sorted(payload["themes"], key=lambda item: item["theme_id"]):
        print(f"{theme['theme_id']}  n={theme['size']}  {theme['label']}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
