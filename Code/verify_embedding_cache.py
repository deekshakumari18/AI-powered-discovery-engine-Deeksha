"""Verify the FastEmbed ONNX model cache used at build time and at runtime.

Build step (`python Code/verify_embedding_cache.py`):
  1. downloads the exact model into FASTEMBED_CACHE_PATH,
  2. re-loads it in a fresh process with `local_files_only=True`, which fails
     unless every model file is already present in that cache,
  3. encodes one query and checks dimension, normalisation and that PyTorch is
     not imported.

The runtime uses the same FASTEMBED_CACHE_PATH, so the first chat request never
has to download the model.

Run `python Code/verify_embedding_cache.py --check-only` to skip the download.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = PROJECT_ROOT / ".fastembed_cache"
EXPECTED_DIM = 384


def cache_dir() -> Path:
    return Path(os.environ.get("FASTEMBED_CACHE_PATH", str(DEFAULT_CACHE)))


def download() -> None:
    from fastembed import TextEmbedding

    TextEmbedding(
        model_name=MODEL_ID,
        cache_dir=str(cache_dir()),
        threads=1,
        providers=["CPUExecutionProvider"],
        batch_size=1,
    )


def check_local() -> None:
    import numpy as np
    from fastembed import TextEmbedding

    model = TextEmbedding(
        model_name=MODEL_ID,
        cache_dir=str(cache_dir()),
        threads=1,
        providers=["CPUExecutionProvider"],
        batch_size=1,
        local_files_only=True,
    )
    vector = np.asarray(
        list(model.embed(["cache verification query"], batch_size=1))[0],
        dtype=np.float32,
    )
    if vector.shape != (EXPECTED_DIM,):
        raise SystemExit(f"Unexpected embedding shape: {vector.shape}")
    norm = float(np.linalg.norm(vector))
    if abs(norm - 1.0) > 1e-4:
        raise SystemExit(f"Embedding is not L2-normalised: norm={norm}")
    if "torch" in sys.modules or "sentence_transformers" in sys.modules:
        raise SystemExit("PyTorch leaked into the ONNX embedding path")


def cache_size_mb() -> float:
    total = 0
    for root, _dirs, files in os.walk(cache_dir()):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return round(total / 1e6, 1)


def main() -> None:
    check_only = "--check-only" in sys.argv
    if check_only:
        check_local()
        print(
            f"Embedding cache OK: model={MODEL_ID} dir={cache_dir()} "
            f"size={cache_size_mb()}MB local_files_only=True"
        )
        return
    download()
    env = dict(os.environ)
    env["FASTEMBED_CACHE_PATH"] = str(cache_dir())
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--check-only"],
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit("Model cache verification failed")
    print(
        f"Embedding cache OK: model={MODEL_ID} dir={cache_dir()} "
        f"size={cache_size_mb()}MB local_files_only=True"
    )


if __name__ == "__main__":
    main()
