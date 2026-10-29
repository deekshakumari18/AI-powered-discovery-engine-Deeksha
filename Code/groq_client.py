"""Call Groq with the single key and model in .env."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 45.0


def _timeout_seconds() -> float:
    raw = os.environ.get("GROQ_TIMEOUT", "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning("groq: invalid GROQ_TIMEOUT=%r, using %.1fs", raw, DEFAULT_TIMEOUT_SECONDS)
        return DEFAULT_TIMEOUT_SECONDS
    if value <= 0:
        logger.warning("groq: non-positive GROQ_TIMEOUT=%r, using %.1fs", raw, DEFAULT_TIMEOUT_SECONDS)
        return DEFAULT_TIMEOUT_SECONDS
    return value


def generate(prompt: str) -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        logger.error("groq: GROQ_API_KEY is not set")
        raise RuntimeError("Set GROQ_API_KEY in the project .env file.")
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b").strip() or "openai/gpt-oss-20b"
    timeout = _timeout_seconds()

    from groq import Groq

    logger.info(
        "groq: request start model=%s prompt_chars=%d timeout=%.1fs",
        model,
        len(prompt),
        timeout,
    )
    started = time.perf_counter()
    try:
        client = Groq(api_key=key, timeout=timeout)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:
        logger.exception(
            "groq: request failed after %.2fs",
            time.perf_counter() - started,
        )
        raise

    elapsed = time.perf_counter() - started
    text = (response.choices[0].message.content or "").strip()
    if not text:
        logger.error("groq: empty answer after %.2fs", elapsed)
        raise RuntimeError("Groq returned an empty answer.")
    logger.info(
        "groq: request completed in %.2fs (%d chars)",
        elapsed,
        len(text),
    )
    return text
