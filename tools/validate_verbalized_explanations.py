from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = ROOT / "revision_outputs" / "explanation_records.jsonl"

SUPPORT_WORDS = [
    "benefit",
    "support",
    "supports",
    "supporting",
    "positive",
    "closer",
    "safer",
    "better",
    "ưu tiên",
    "lợi ích",
    "an toàn",
    "tốt hơn",
]
TRADEOFF_WORDS = [
    "trade-off",
    "tradeoff",
    "limitation",
    "negative",
    "drawback",
    "less",
    "further",
    "worse",
    "hạn chế",
    "xa",
    "kém",
    "bất lợi",
]
STOCHASTIC_WORDS = [
    "stochastic",
    "random",
    "sample",
    "sampling",
    "probability",
    "probabilistic",
    "roulette",
    "ngẫu nhiên",
    "xác suất",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append({"__load_error__": f"line {line_no}: {exc}"})
                continue
            rows.append(row)
    return rows


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).lower()


def as_output_text(output: dict[str, Any] | str | None, fallback_template: str | None = None) -> tuple[str, list[str]]:
    issues: list[str] = []
    if output is None:
        return normalize_text(fallback_template), issues
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
            output = parsed
        except json.JSONDecodeError:
            return normalize_text(output), issues
    if not isinstance(output, dict):
        return "", ["LLM output is neither a string nor a JSON object"]
    expected = {"local_explanation", "evidence_used", "limitations"}
    missing = sorted(expected - set(output))
    extra = sorted(set(output) - expected)
    if missing:
        issues.append(f"LLM output missing required fields: {', '.join(missing)}")
    if extra:
        issues.append(f"LLM output has extra fields: {', '.join(extra)}")
    if "local_explanation" in output and not isinstance(output["local_explanation"], str):
        issues.append("local_explanation must be a string")
    if "evidence_used" in output and not isinstance(output["evidence_used"], list):
        issues.append("evidence_used must be an array")
    if "limitations" in output and not isinstance(output["limitations"], list):
        issues.append("limitations must be an array")
    for field in expected:
        if field in output and output[field] is None:
            issues.append(f"{field} must not be null")
    return normalize_text(output.get("local_explanation", "")), issues


def mentions(text: str, value: Any) -> bool:
    if value is None:
        return True
    raw = str(value).strip()
    if raw == "":
        return True
    return raw.lower() in text


def contributor_names(record: dict[str, Any], field: str) -> list[str]:
    return [str(item.get("name", "")).lower() for item in record.get(field, []) if item.get("name")]


def detect_polarity_inversion(text: str, record: dict[str, Any]) -> list[str]:
    issues = []
    positive = contributor_names(record, "positive_contributors")
    negative = contributor_names(record, "negative_contributors")
    support_context = "|".join(re.escape(word) for word in SUPPORT_WORDS)
    tradeoff_context = "|".join(re.escape(word) for word in TRADEOFF_WORDS)

    for name in positive:
        if name in text:
            pattern = rf"({tradeoff_context}).{{0,40}}\b{re.escape(name)}\b|\b{re.escape(name)}\b.{{0,40}}({tradeoff_context})"
            if re.search(pattern, text):
                issues.append(f"positive contributor described as trade-off: {name}")
    for name in negative:
        if name in text:
            pattern = rf"({support_context}).{{0,40}}\b{re.escape(name)}\b|\b{re.escape(name)}\b.{{0,40}}({support_context})"
            if re.search(pattern, text):
                issues.append(f"negative contributor described as support: {name}")
    return issues


def detect_unsupported_contributors(text: str, record: dict[str, Any]) -> list[str]:
    allowed = set(contributor_names(record, "positive_contributors") + contributor_names(record, "negative_contributors"))
    known = {"turn", "goal", "blocked", "safe", "open-4", "block-open-4", "center-dist", "center"}
    issues = []
    for name in sorted(known - allowed):
        if re.search(rf"\b{re.escape(name)}\b", text):
            issues.append(f"unsupported contributor mentioned: {name}")
    return issues


def check_numeric_signs(text: str, record: dict[str, Any]) -> list[str]:
    issues = []
    contributors = record.get("positive_contributors", []) + record.get("negative_contributors", [])
    for item in contributors:
        value = item.get("value")
        name = item.get("name")
        if value is None or name is None:
            continue
        number = float(value)
        signed = f"{number:+.2f}".lower()
        unsigned = f"{abs(number):.2f}".lower()
        if unsigned in text and signed not in text:
            if number < 0 and f"+{unsigned}" in text:
                issues.append(f"numeric sign inverted for {name}: expected negative")
            if number > 0 and f"-{unsigned}" in text:
                issues.append(f"numeric sign inverted for {name}: expected positive")
    return issues


def check_stochasticity(text: str, record: dict[str, Any]) -> list[str]:
    mechanism = record.get("selection_mechanism")
    selected_probability = record.get("selected_probability")
    if mechanism not in {"RWS", "BTMM+RWS"}:
        return []
    if selected_probability is None or selected_probability >= 0.05:
        return []
    if any(word in text for word in STOCHASTIC_WORDS):
        return []
    return ["low-probability stochastic selection is not described as stochastic/probabilistic"]


def validate_pair(record: dict[str, Any], output: dict[str, Any] | str | None) -> list[str]:
    if "__load_error__" in record:
        return [record["__load_error__"]]
    text, issues = as_output_text(output, record.get("template_explanation"))
    if not text:
        issues.append("local_explanation/template text is empty")
        return issues
    if not mentions(text, record.get("selected_action")):
        issues.append(f"selected action not mentioned: {record.get('selected_action')}")
    if record.get("alternative_action") is not None and not mentions(text, record.get("alternative_action")):
        issues.append(f"alternative action not mentioned: {record.get('alternative_action')}")
    issues.extend(detect_polarity_inversion(text, record))
    issues.extend(detect_unsupported_contributors(text, record))
    issues.extend(check_numeric_signs(text, record))
    issues.extend(check_stochasticity(text, record))
    return issues


def issue_category(issue: str) -> str:
    if issue.startswith("line ") or "neither a string nor a JSON object" in issue:
        return "schema or JSON parsing failure"
    if "missing required fields" in issue:
        return "missing required field"
    if "extra fields" in issue or "must be" in issue:
        return "schema or JSON parsing failure"
    if issue.startswith("selected action not mentioned"):
        return "selected-action mismatch"
    if issue.startswith("alternative action not mentioned"):
        return "alternative-action mismatch"
    if issue.startswith("unsupported contributor mentioned"):
        return "unsupported contributor"
    if "described as trade-off" in issue or "described as support" in issue:
        return "contributor-polarity mismatch"
    if "numeric sign" in issue:
        return "numerical-sign mismatch"
    if "stochastic" in issue or "probabilistic" in issue:
        return "stochasticity inconsistency"
    return "other"


def output_key(row: dict[str, Any], index: int) -> str:
    if row.get("record_id"):
        return str(row["record_id"])
    case_study = row.get("case_study")
    source_file = row.get("source_file")
    source_row_id = row.get("source_row_id")
    if case_study is not None and source_file is not None and source_row_id is not None:
        return f"{case_study}|{source_file}|{source_row_id}"
    return str(source_row_id or index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LLM verbalizations against XRL explanation records.")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS, help="Explanation records JSONL.")
    parser.add_argument(
        "--llm-outputs",
        type=Path,
        default=None,
        help="Optional JSONL with LLM outputs. If omitted, validates template_explanation fields in records.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "revision_outputs" / "llm_validation_report.json",
        help="Validation report JSON path.",
    )
    args = parser.parse_args()

    records = load_jsonl(args.records)
    outputs_by_key: dict[str, Any] = {}
    if args.llm_outputs:
        for idx, row in enumerate(load_jsonl(args.llm_outputs), start=1):
            outputs_by_key[output_key(row, idx)] = row.get("llm_output", row)

    results = []
    for idx, record in enumerate(records, start=1):
        key = output_key(record, idx)
        output = outputs_by_key.get(key) if args.llm_outputs else None
        issues = validate_pair(record, output)
        categories = sorted({issue_category(issue) for issue in issues})
        results.append(
            {
                "record_key": key,
                "case_study": record.get("case_study"),
                "source_file": record.get("source_file"),
                "source_row_id": record.get("source_row_id"),
                "passed": len(issues) == 0,
                "issues": issues,
                "error_categories": categories,
            }
        )

    summary = {
        "records_checked": len(results),
        "passed": sum(1 for item in results if item["passed"]),
        "failed": sum(1 for item in results if not item["passed"]),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
