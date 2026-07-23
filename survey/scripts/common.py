from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw_exports"
OUT_DIR = ROOT / "out"
LEDGER = ROOT / "ledger.csv"
EXTRACTION = ROOT / "extraction.csv"


SCREENING_COLUMNS = [
    "record_id",
    "title",
    "year",
    "venue",
    "doi",
    "ta_rater_A",
    "ta_rater_B",
    "ta_final",
    "ft_rater_A",
    "ft_rater_B",
    "ft_final",
    "exclusion_code",
    "note",
]

EXTRACTION_COLUMNS = [
    "record_id",
    "cite_key",
    "year",
    "venue",
    "decision_mechanism",
    "scope_intrinsic_posthoc",
    "scope_local_global",
    "artifact",
    "eval_metric_type",
    "faithfulness_measured",
    "human_study",
    "selection_rule_explained",
    "code_available",
    "domain",
]

DECISIONS = {"include", "exclude"}
EXCLUSION_CODES = {"", "E1", "E2", "E3", "E4"}
METHOD_FAMILIES = {"Decompose", "Score", "Select", "Explain", "Validate"}
INTRINSIC_POSTHOC = {"intrinsic", "post-hoc", "mixed"}
LOCAL_GLOBAL = {"local", "global", "both"}
YES_NO = {"yes", "no"}


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(fh)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_doi(value: str) -> str:
    doi = (value or "").strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.strip().rstrip(".")


def normalize_title(value: str) -> str:
    text = (value or "").lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_decision(value: str) -> str:
    value = (value or "").strip().lower()
    aliases = {
        "yes": "include",
        "y": "include",
        "pass": "include",
        "keep": "include",
        "included": "include",
        "no": "exclude",
        "n": "exclude",
        "fail": "exclude",
        "drop": "exclude",
        "excluded": "exclude",
    }
    return aliases.get(value, value)


def require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {path}. {hint}")


def format_errors(errors: list[str], header: str) -> str:
    preview = "\n".join(f"- {item}" for item in errors[:50])
    suffix = "" if len(errors) <= 50 else f"\n... {len(errors) - 50} more"
    return f"{header}\n{preview}{suffix}"
