from __future__ import annotations

import argparse
import csv
import hashlib
import re
from difflib import SequenceMatcher
from pathlib import Path

from common import OUT_DIR, RAW_DIR, ensure_dirs, normalize_doi, normalize_title, write_csv, write_json


FIELDS = ["record_id", "source_db", "title", "authors", "year", "venue", "doi", "url", "abstract", "dup_of"]


def bib_entries(text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for match in re.finditer(r"@\w+\s*\{\s*([^,]+),(.*?)(?=\n@\w+\s*\{|\Z)", text, re.S | re.I):
        body = match.group(2)
        row: dict[str, str] = {}
        for field in ["title", "author", "year", "journal", "booktitle", "doi", "url", "abstract"]:
            found = re.search(rf"\b{field}\s*=\s*[\{{\"](.*?)[\}}\"]\s*,?\s*(?=\n\s*\w+\s*=|\n\s*\}}|\Z)", body, re.S | re.I)
            if found:
                row[field] = re.sub(r"\s+", " ", found.group(1)).strip()
        entries.append(row)
    return entries


def ris_entries(text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    current: dict[str, list[str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if len(line) < 6 or line[2:6] != "  - ":
            continue
        tag, value = line[:2], line[6:].strip()
        if tag == "TY":
            current = {}
        elif tag == "ER":
            entries.append({key: "; ".join(values) for key, values in current.items()})
            current = {}
        else:
            current.setdefault(tag, []).append(value)
    if current:
        entries.append({key: "; ".join(values) for key, values in current.items()})
    return entries


def csv_entries(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [{(key or "").strip().lower(): (value or "").strip() for key, value in row.items()} for row in csv.DictReader(fh)]


def pick(row: dict[str, str], *names: str) -> str:
    lowered = {key.lower(): value for key, value in row.items()}
    for name in names:
        if lowered.get(name.lower()):
            return lowered[name.lower()].strip()
    return ""


def normalize_row(row: dict[str, str], source_db: str) -> dict[str, str]:
    doi = normalize_doi(pick(row, "doi", "DO", "DI"))
    title = pick(row, "title", "TI", "T1", "article title")
    authors = pick(row, "authors", "author", "AU", "authors/authors")
    year = pick(row, "year", "PY", "Y1", "publication year")
    venue = pick(row, "venue", "journal", "booktitle", "JO", "JF", "T2", "publication title")
    url = pick(row, "url", "UR", "link")
    abstract = pick(row, "abstract", "AB")
    if doi:
        record_id = "doi_" + re.sub(r"[^a-z0-9]+", "_", doi.lower()).strip("_")
    else:
        digest = hashlib.sha1(normalize_title(title).encode("utf-8")).hexdigest()[:12]
        record_id = "title_" + digest
    return {
        "record_id": record_id,
        "source_db": source_db,
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue,
        "doi": doi,
        "url": url,
        "abstract": abstract,
        "dup_of": "",
    }


def load_raw_file(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if path.suffix.lower() == ".bib":
        return bib_entries(text)
    if path.suffix.lower() == ".ris":
        return ris_entries(text)
    if path.suffix.lower() == ".csv":
        return csv_entries(path)
    return []


def similarity(left: str, right: str) -> float:
    try:
        from rapidfuzz.fuzz import ratio

        return ratio(left, right) / 100.0
    except Exception:
        return SequenceMatcher(None, left, right).ratio()


def mark_duplicates(rows: list[dict[str, str]]) -> None:
    seen_doi: dict[str, str] = {}
    retained_titles: list[tuple[str, str]] = []
    for row in rows:
        doi = normalize_doi(row["doi"])
        title_norm = normalize_title(row["title"])
        if doi and doi in seen_doi:
            row["dup_of"] = seen_doi[doi]
            continue
        title_dup = ""
        if title_norm:
            for retained_title, retained_id in retained_titles:
                if similarity(title_norm, retained_title) >= 0.95:
                    title_dup = retained_id
                    break
        if title_dup:
            row["dup_of"] = title_dup
            continue
        if doi:
            seen_doi[doi] = row["record_id"]
        if title_norm:
            retained_titles.append((title_norm, row["record_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and deduplicate raw survey exports.")
    parser.parse_args()
    ensure_dirs()
    files = sorted(path for path in RAW_DIR.iterdir() if path.suffix.lower() in {".bib", ".ris", ".csv"})
    if not files:
        raise SystemExit(f"No .bib/.ris/.csv files found in {RAW_DIR}. Export the databases first.")

    rows: list[dict[str, str]] = []
    per_db: dict[str, int] = {}
    for path in files:
        source_db = path.stem
        entries = load_raw_file(path)
        per_db[source_db] = len(entries)
        for entry in entries:
            rows.append(normalize_row(entry, source_db))

    mark_duplicates(rows)
    warnings = [row["record_id"] for row in rows if not row["doi"] and not row["year"]]
    for record_id in warnings:
        print(f"WARNING: {record_id} is missing both DOI and year")

    n_duplicates = sum(1 for row in rows if row["dup_of"])
    counts = {
        "per_db": per_db,
        "n_identified": len(rows),
        "n_duplicates": n_duplicates,
        "n_after_dedup": len(rows) - n_duplicates,
    }
    write_csv(OUT_DIR / "records.csv", rows, FIELDS)
    write_json(OUT_DIR / "counts_by_source.json", counts)
    print(f"Wrote {OUT_DIR / 'records.csv'}")
    print(f"Wrote {OUT_DIR / 'counts_by_source.json'}")


if __name__ == "__main__":
    main()
