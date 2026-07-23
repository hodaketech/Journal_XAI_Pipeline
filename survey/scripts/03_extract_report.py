from __future__ import annotations

import argparse
from collections import Counter

from common import (
    EXTRACTION,
    EXTRACTION_COLUMNS,
    INTRINSIC_POSTHOC,
    LEDGER,
    LOCAL_GLOBAL,
    METHOD_FAMILIES,
    OUT_DIR,
    YES_NO,
    ensure_dirs,
    format_errors,
    normalize_decision,
    read_csv,
    read_json,
    require_file,
    write_csv,
    write_json,
)


def included_records() -> list[dict[str, str]]:
    require_file(LEDGER, "Run survey/scripts/02_screen_report.py and fill ledger.csv first.")
    by_id = {row["record_id"]: row for row in read_csv(OUT_DIR / "records.csv") if not row.get("dup_of")}
    included = []
    for row in read_csv(LEDGER):
        if normalize_decision(row.get("ta_final", "")) == "include" and normalize_decision(row.get("ft_final", "")) == "include":
            base = by_id.get(row["record_id"], {})
            included.append({"record_id": row["record_id"], "year": base.get("year", row.get("year", "")), "venue": base.get("venue", row.get("venue", ""))})
    return included


def create_extraction() -> None:
    rows = included_records()
    if not rows:
        raise SystemExit("No included records found. Complete ledger.csv and screening_summary.json first.")
    write_csv(EXTRACTION, rows, EXTRACTION_COLUMNS)
    print(f"Created blank extraction sheet: {EXTRACTION}")
    print("Fill all coding fields before rerunning this script.")


def split_methods(value: str) -> list[str]:
    return [item.strip() for item in value.replace("|", ";").replace(",", ";").split(";") if item.strip()]


def validate(rows: list[dict[str, str]], expected_ids: set[str]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    for line_no, row in enumerate(rows, start=2):
        rid = row.get("record_id", "")
        seen_ids.add(rid)
        if rid not in expected_ids:
            errors.append(f"CSV line {line_no}: record_id {rid!r} is not an included full-text record")
        for field in EXTRACTION_COLUMNS:
            if not row.get(field, "").strip():
                errors.append(f"{rid}: {field} is blank at CSV line {line_no}")
        methods = split_methods(row.get("decision_mechanism", ""))
        bad_methods = [item for item in methods if item not in METHOD_FAMILIES]
        if not methods or bad_methods:
            errors.append(f"{rid}: decision_mechanism must be a subset of {sorted(METHOD_FAMILIES)}")
        if row.get("scope_intrinsic_posthoc", "") not in INTRINSIC_POSTHOC:
            errors.append(f"{rid}: scope_intrinsic_posthoc must be one of {sorted(INTRINSIC_POSTHOC)}")
        if row.get("scope_local_global", "") not in LOCAL_GLOBAL:
            errors.append(f"{rid}: scope_local_global must be one of {sorted(LOCAL_GLOBAL)}")
        for field in ["faithfulness_measured", "human_study", "selection_rule_explained", "code_available"]:
            if row.get(field, "") not in YES_NO:
                errors.append(f"{rid}: {field} must be yes or no")
    missing = expected_ids - seen_ids
    extra = seen_ids - expected_ids
    if missing:
        errors.append(f"Missing included record_id values: {', '.join(sorted(missing))}")
    if extra:
        errors.append(f"Unexpected record_id values: {', '.join(sorted(extra))}")
    return errors


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    method_counts: Counter[str] = Counter()
    intrinsic_counts: Counter[str] = Counter()
    local_counts: Counter[str] = Counter()
    faithfulness_counts: Counter[str] = Counter()
    human_counts: Counter[str] = Counter()
    selection_counts: Counter[str] = Counter()
    code_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()

    for row in rows:
        for method in split_methods(row["decision_mechanism"]):
            method_counts[method] += 1
        intrinsic_counts[row["scope_intrinsic_posthoc"]] += 1
        local_counts[row["scope_local_global"]] += 1
        faithfulness_counts[row["faithfulness_measured"]] += 1
        human_counts[row["human_study"]] += 1
        selection_counts[row["selection_rule_explained"]] += 1
        code_counts[row["code_available"]] += 1
        domain_counts[row["domain"]] += 1

    return {
        "n_records": len(rows),
        "method_family_counts": {key: method_counts.get(key, 0) for key in ["Decompose", "Score", "Select", "Explain", "Validate"]},
        "scope_intrinsic_posthoc_counts": dict(sorted(intrinsic_counts.items())),
        "scope_local_global_counts": dict(sorted(local_counts.items())),
        "faithfulness_measured": faithfulness_counts.get("yes", 0),
        "faithfulness_asserted_or_not_measured": faithfulness_counts.get("no", 0),
        "human_study": human_counts.get("yes", 0),
        "selection_rule_explained": selection_counts.get("yes", 0),
        "code_available": code_counts.get("yes", 0),
        "domain_counts": dict(sorted(domain_counts.items())),
    }


def table_b2_rows(stats: dict[str, object]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group, values in [
        ("Method family", stats["method_family_counts"]),
        ("Intrinsic/post-hoc", stats["scope_intrinsic_posthoc_counts"]),
        ("Local/global", stats["scope_local_global_counts"]),
        ("Domain", stats["domain_counts"]),
    ]:
        for label, count in dict(values).items():
            rows.append({"group": group, "category": label, "count": count})
    rows.extend(
        [
            {"group": "Evaluation", "category": "Faithfulness measured", "count": stats["faithfulness_measured"]},
            {"group": "Evaluation", "category": "Faithfulness asserted/not measured", "count": stats["faithfulness_asserted_or_not_measured"]},
            {"group": "Evaluation", "category": "Human study", "count": stats["human_study"]},
            {"group": "Reproducibility", "category": "Selection rule explained", "count": stats["selection_rule_explained"]},
            {"group": "Reproducibility", "category": "Code available", "count": stats["code_available"]},
        ]
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate extraction sheet and compute derived survey statistics.")
    parser.parse_args()
    ensure_dirs()
    require_file(OUT_DIR / "screening_summary.json", "Run survey/scripts/02_screen_report.py after completing ledger.csv first.")
    if not EXTRACTION.exists():
        create_extraction()
        return
    expected = {row["record_id"] for row in included_records()}
    rows = read_csv(EXTRACTION)
    errors = validate(rows, expected)
    if errors:
        raise SystemExit(format_errors(errors, "Extraction sheet is incomplete or invalid; no derived stats were written."))
    stats = summarize(rows)
    write_json(OUT_DIR / "derived_stats.json", stats)
    write_csv(OUT_DIR / "table_B1.csv", rows, EXTRACTION_COLUMNS)
    write_csv(OUT_DIR / "table_B2.csv", table_b2_rows(stats), ["group", "category", "count"])
    print(f"Wrote {OUT_DIR / 'derived_stats.json'}")
    print(f"Wrote {OUT_DIR / 'table_B1.csv'}")
    print(f"Wrote {OUT_DIR / 'table_B2.csv'}")


if __name__ == "__main__":
    main()
