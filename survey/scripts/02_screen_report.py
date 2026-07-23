from __future__ import annotations

import argparse
from collections import Counter

from common import (
    DECISIONS,
    EXCLUSION_CODES,
    LEDGER,
    OUT_DIR,
    SCREENING_COLUMNS,
    ensure_dirs,
    format_errors,
    normalize_decision,
    read_csv,
    read_json,
    require_file,
    write_csv,
    write_json,
)


def cohen_kappa(left: list[str], right: list[str]) -> float | None:
    if not left or len(left) != len(right):
        return None
    labels = sorted(set(left) | set(right))
    observed = sum(1 for a, b in zip(left, right) if a == b) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum((left_counts[label] / len(left)) * (right_counts[label] / len(right)) for label in labels)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def agreement(left: list[str], right: list[str]) -> float | None:
    if not left:
        return None
    return sum(1 for a, b in zip(left, right) if a == b) / len(left)


def retained_records() -> list[dict[str, str]]:
    require_file(OUT_DIR / "records.csv", "Run survey/scripts/01_ingest.py first.")
    return [row for row in read_csv(OUT_DIR / "records.csv") if not row.get("dup_of")]


def create_ledger() -> None:
    rows = [
        {
            "record_id": row["record_id"],
            "title": row["title"],
            "year": row["year"],
            "venue": row["venue"],
            "doi": row["doi"],
        }
        for row in retained_records()
    ]
    write_csv(LEDGER, rows, SCREENING_COLUMNS)
    print(f"Created blank ledger: {LEDGER}")
    print("Fill the rater/final decision columns before rerunning this script.")


def last_search_date(cli_value: str) -> str:
    if cli_value:
        return cli_value
    path = OUT_DIR.parent / "last_search_date.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    raise SystemExit("Missing last search date. Pass --last-search-date YYYY-MM-DD or create survey/last_search_date.txt.")


def validate_and_summarize(search_date: str) -> dict[str, object]:
    counts = read_json(OUT_DIR / "counts_by_source.json")
    rows = read_csv(LEDGER)
    errors: list[str] = []
    expected_ids = {row["record_id"] for row in retained_records()}
    ledger_ids = {row.get("record_id", "") for row in rows}
    missing_ids = expected_ids - ledger_ids
    extra_ids = ledger_ids - expected_ids
    if missing_ids:
        errors.append(f"ledger.csv is missing deduplicated record_id values: {', '.join(sorted(missing_ids))}")
    if extra_ids:
        errors.append(f"ledger.csv contains unknown record_id values: {', '.join(sorted(extra_ids))}")
    ta_a: list[str] = []
    ta_b: list[str] = []
    ft_a: list[str] = []
    ft_b: list[str] = []
    disagreements: list[dict[str, str]] = []
    exclusions: Counter[str] = Counter()

    for line_no, row in enumerate(rows, start=2):
        rid = row.get("record_id", f"line {line_no}")
        ta_values = {name: normalize_decision(row.get(name, "")) for name in ["ta_rater_A", "ta_rater_B", "ta_final"]}
        for name, value in ta_values.items():
            if value not in DECISIONS:
                errors.append(f"{rid}: {name} is blank or invalid at CSV line {line_no}")
        if all(value in DECISIONS for value in ta_values.values()):
            ta_a.append(ta_values["ta_rater_A"])
            ta_b.append(ta_values["ta_rater_B"])
            if ta_values["ta_rater_A"] != ta_values["ta_rater_B"]:
                disagreements.append({"record_id": rid, "stage": "title_abstract", "rater_A": ta_values["ta_rater_A"], "rater_B": ta_values["ta_rater_B"]})

        if ta_values.get("ta_final") != "include":
            continue

        ft_values = {name: normalize_decision(row.get(name, "")) for name in ["ft_rater_A", "ft_rater_B", "ft_final"]}
        for name, value in ft_values.items():
            if value not in DECISIONS:
                errors.append(f"{rid}: {name} is blank or invalid at CSV line {line_no}")
        if all(value in DECISIONS for value in ft_values.values()):
            ft_a.append(ft_values["ft_rater_A"])
            ft_b.append(ft_values["ft_rater_B"])
            if ft_values["ft_rater_A"] != ft_values["ft_rater_B"]:
                disagreements.append({"record_id": rid, "stage": "full_text", "rater_A": ft_values["ft_rater_A"], "rater_B": ft_values["ft_rater_B"]})
        code = (row.get("exclusion_code") or "").strip().upper()
        if code not in EXCLUSION_CODES:
            errors.append(f"{rid}: exclusion_code must be blank or E1/E2/E3/E4 at CSV line {line_no}")
        if ft_values.get("ft_final") == "exclude":
            if not code:
                errors.append(f"{rid}: ft_final=exclude requires exclusion_code at CSV line {line_no}")
            else:
                exclusions[code] += 1

    if errors:
        raise SystemExit(format_errors(errors, "Ledger is incomplete or invalid; no summary was written."))

    n_ta_pass = sum(1 for row in rows if normalize_decision(row.get("ta_final", "")) == "include")
    n_ft_pass = sum(1 for row in rows if normalize_decision(row.get("ta_final", "")) == "include" and normalize_decision(row.get("ft_final", "")) == "include")
    summary = {
        "n_identified": counts["n_identified"],
        "n_duplicates": counts["n_duplicates"],
        "n_after_dedup": counts["n_after_dedup"],
        "per_db": counts["per_db"],
        "n_title_abstract_pass": n_ta_pass,
        "n_fulltext_pass": n_ft_pass,
        "exclusions": {code: exclusions.get(code, 0) for code in ["E1", "E2", "E3", "E4"]},
        "kappa_ta": cohen_kappa(ta_a, ta_b),
        "kappa_ft": cohen_kappa(ft_a, ft_b),
        "agreement_ta": agreement(ta_a, ta_b),
        "agreement_ft": agreement(ft_a, ft_b),
        "last_search_date": search_date,
    }
    write_csv(OUT_DIR / "screening_disagreements.csv", disagreements, ["record_id", "stage", "rater_A", "rater_B"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate screening ledger and compute PRISMA counts.")
    parser.add_argument("--last-search-date", default="", help="Evidence-backed last search date, e.g. 2026-07-22.")
    parser.add_argument("--rebuild-ledger", action="store_true", help="Overwrite ledger.csv from current deduplicated records.")
    args = parser.parse_args()
    ensure_dirs()
    require_file(OUT_DIR / "counts_by_source.json", "Run survey/scripts/01_ingest.py first.")
    if args.rebuild_ledger or not LEDGER.exists():
        create_ledger()
        return
    summary = validate_and_summarize(last_search_date(args.last_search_date))
    write_json(OUT_DIR / "screening_summary.json", summary)
    print(f"Wrote {OUT_DIR / 'screening_summary.json'}")


if __name__ == "__main__":
    main()
