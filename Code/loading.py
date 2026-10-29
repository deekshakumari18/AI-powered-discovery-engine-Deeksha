"""Phase 1 — Data loading.

Reads Groww_SIP_Investor_Journey_Review_Corpus.docx from Data/raw.
Writes citable text units (no PII) to Data/raw/loaded.json.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path

from docx import Document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "Data" / "raw"
ALLOWED_CORPUS = "Groww_SIP_Investor_Journey_Review_Corpus.docx"
LOADED_PATH = RAW_DIR / "loaded.json"

RECORD_START = re.compile(r"^Record\s+\d+\s*:", re.IGNORECASE)
SEPARATOR = re.compile(r"^─{8,}$")
SOURCE_LINK = re.compile(r"Source link:\s*(\S+)", re.IGNORECASE)
STORE_LINK = re.compile(
    r"https?://(?:play\.google\.com|itunes\.apple\.com|apps\.apple\.com)\S*",
    re.IGNORECASE,
)
RECORD_DATE = re.compile(
    r"Date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:\.]+Z|[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}(?:\s+\d{1,2}:\d{2}:\d{2})?)",
    re.IGNORECASE,
)

EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
# Spaced 4-4-4 only, or labeled. Do not eat 10–12 digit review IDs.
AADHAAR = re.compile(
    r"(?:(?:aadhaar|uidai|uid)\s*[:\-]?\s*)?(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)",
    re.IGNORECASE,
)
# +91, spaced/dashed Indian mobiles, or phone/mobile labeled. Never bare 10-digit IDs.
PHONE = re.compile(
    r"(?:\+91[\s\-]*[6-9]\d{9}\b"
    r"|(?:phone|mobile|mob|whatsapp)\s*[:\-]?\s*[6-9]\d{9}\b"
    r"|[6-9]\d{4}[\s\-]\d{5}\b)",
    re.IGNORECASE,
)
OTP = re.compile(
    r"\b(?:otp|one[-\s]?time(?:\s+password)?)\s*[:\-]?\s*\d{4,8}\b",
    re.IGNORECASE,
)
# Only account/IFSC *numbers*, not the words account / accessible.
ACCOUNT = re.compile(
    r"\b(?:account|a/?c|acct)\s*(?:no\.?|number|#)\s*[:\-]?\s*\d{8,18}\b"
    r"|\b[A-Z]{4}0[A-Z0-9]{6}\b",
    re.IGNORECASE,
)
HOLD_ID = re.compile(r"\bID:\s*[A-Za-z0-9\-]+", re.IGNORECASE)
HOLD_URL = re.compile(r"https?://\S+", re.IGNORECASE)

DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
)


def redact_pii(text: str) -> str:
    held: list[str] = []

    def _hold(match: re.Match) -> str:
        held.append(match.group(0))
        return f"__HOLD_{len(held) - 1}__"

    text = HOLD_URL.sub(_hold, text)
    text = HOLD_ID.sub(_hold, text)
    text = EMAIL.sub("[EMAIL]", text)
    text = PAN.sub("[PAN]", text)
    text = OTP.sub("[OTP]", text)
    text = ACCOUNT.sub("[ACCOUNT]", text)
    text = PHONE.sub("[PHONE]", text)
    text = AADHAAR.sub("[AADHAAR]", text)
    for i, original in enumerate(held):
        text = text.replace(f"__HOLD_{i}__", original)
    return text


def _parse_record_date(raw: str) -> datetime | None:
    value = raw.strip().replace("Z", "")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        pass
    if "." in value and "T" in value:
        value = value.split(".")[0]
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _review_date(text: str) -> str | None:
    match = RECORD_DATE.search(text)
    if not match:
        return None
    parsed = _parse_record_date(match.group(1))
    if parsed is None:
        return None
    return parsed.date().isoformat()


def _is_store_link(url: str) -> bool:
    return STORE_LINK.search(url) is not None


def _strip_store_links(text: str) -> str:
    kept = []
    for line in text.splitlines():
        if line.strip().lower().startswith("source link:") and _is_store_link(line):
            continue
        kept.append(STORE_LINK.sub("", line).rstrip())
    return "\n".join(kept).strip()


def _citation_url(text: str) -> str | None:
    match = SOURCE_LINK.search(text)
    if not match:
        return None
    url = match.group(1).rstrip(").,;")
    if not (url.startswith("http://") or url.startswith("https://")):
        return None
    if _is_store_link(url):
        return None
    return url


TRANSLATION = re.compile(
    r"English translation:\s*(.+?)(?:\nSource link:|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _original_body(text: str) -> str:
    marker = "Original text:"
    start = text.find(marker)
    if start < 0:
        return text.strip()
    body = text[start + len(marker) :]
    end = body.rfind("Source link:")
    if end >= 0:
        body = body[:end]
    return body.strip()


def _source_type(section: str | None, text: str) -> str:
    sec = (section or "").lower()
    if "youtube" in sec:
        return "youtube"
    if "play store" in sec:
        return "play_store"
    if "app store" in sec:
        return "app_store"
    if "reddit" in sec:
        return "reddit"
    first = text.splitlines()[0].lower() if text else ""
    if "youtube" in first or "comment by @" in first:
        return "youtube"
    if "play store" in first:
        return "play_store"
    if "app store" in first:
        return "app_store"
    return "other"


def enrich_unit(unit: dict) -> dict:
    text = unit.get("text") or ""
    enriched = dict(unit)
    enriched["source_type"] = _source_type(unit.get("section"), text)
    trans = None
    match = TRANSLATION.search(text)
    if match:
        trans = match.group(1).strip()
    enriched["english_translation"] = trans
    body = _original_body(text)
    enriched["embed_text"] = f"{body}\n{trans}" if trans else body
    return enriched


def enrich_loaded(path: Path | None = None) -> Path:
    src = path or LOADED_PATH
    payload = json.loads(src.read_text(encoding="utf-8"))
    payload["units"] = [enrich_unit(unit) for unit in payload["units"]]
    src.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return src


def _flush_record(
    lines: list[str],
    *,
    source_name: str,
    section: str | None,
    ingest_date: str,
    unit_id: int,
) -> dict | None:
    text = _strip_store_links(redact_pii("\n".join(lines).strip()))
    if not text or not RECORD_START.search(text.splitlines()[0]):
        return None
    return {
        "id": f"rec-{unit_id:04d}",
        "text": text,
        "source_name": source_name,
        "section": section,
        "citation_url": _citation_url(text),
        "review_date": _review_date(text),
        "last_updated": ingest_date,
    }


def load_corpus(docx_path: Path | None = None) -> list[dict]:
    path = Path(docx_path) if docx_path else RAW_DIR / ALLOWED_CORPUS
    if path.name != ALLOWED_CORPUS:
        raise ValueError(f"Load only {ALLOWED_CORPUS}; got {path.name}")
    if not path.is_file():
        raise FileNotFoundError(f"Corpus not found: {path}")

    document = Document(str(path))
    ingest_date = date.today().isoformat()

    source_name = path.name
    heading1 = None
    heading2 = None
    buffer: list[str] = []
    units: list[dict] = []

    def section_label() -> str | None:
        parts = [p for p in (heading1, heading2) if p]
        return " | ".join(parts) if parts else None

    def flush() -> None:
        unit = _flush_record(
            buffer,
            source_name=source_name,
            section=section_label(),
            ingest_date=ingest_date,
            unit_id=len(units) + 1,
        )
        buffer.clear()
        if unit:
            units.append(unit)

    for para in document.paragraphs:
        raw = para.text.strip()
        if not raw:
            continue
        style = para.style.name if para.style is not None else "Normal"
        if style == "Heading 1":
            flush()
            heading1 = raw
            heading2 = None
            continue
        if style == "Heading 2":
            flush()
            heading2 = raw
            continue
        if SEPARATOR.match(raw):
            flush()
            continue
        if RECORD_START.match(raw) and buffer:
            flush()
        buffer.append(raw)

    flush()
    return units


def save_loaded(units: list[dict], dest: Path | None = None) -> Path:
    out = dest or LOADED_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus": ALLOWED_CORPUS,
        "unit_count": len(units),
        "ingested_on": date.today().isoformat(),
        "units": units,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    units = load_corpus()
    path = save_loaded(units)
    cited = sum(1 for u in units if u.get("citation_url"))
    print(f"Loaded {len(units)} units ({cited} with citation URL)")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
