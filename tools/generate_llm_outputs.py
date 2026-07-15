from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = ROOT / "revision_outputs" / "explanation_records.jsonl"
DEFAULT_OUTPUT = ROOT / "revision_outputs" / "llm_outputs_ollama_qwen25_7b.jsonl"

PROMPT_VERSION = "xrl-verbalizer-v1.0"

SYSTEM_PROMPT = """You verbalize reinforcement-learning explanation records. Use only supplied evidence. Do not infer hidden causes, recompute scores, change actions, or claim sampled actions are deterministic best actions unless explicitly supported. Return valid JSON only."""

USER_PROMPT_TEMPLATE = """Convert the following XRL explanation record into natural language.

Input record JSON:
{record_json}

Requirements:
1. English, ASCII, one short paragraph.
2. Mention selected_action exactly. Mention alternative_action exactly when not null.
3. Preserve numeric signs for cited values.
4. Positive contributors support the action; negative contributors are trade-offs.
5. Do not invent absent contributors or hidden reasons.
6. For RWS or BTMM+RWS, describe selection as sampled/probabilistic unless highest-scoring evidence is explicit.
7. Return exactly: local_explanation, evidence_used, limitations.
8. No Markdown fences, bullets, or text outside JSON.

Return valid JSON only in this shape:
{{
  "local_explanation": "one short paragraph",
  "evidence_used": ["list of source fields used"],
  "limitations": ["missing or ambiguous fields, if any"]
}}
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_no}: {exc}") from exc
    return rows


def load_existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = output_key(row)
            if key:
                keys.add(key)
    return keys


def output_key(row: dict[str, Any]) -> str:
    if row.get("case_study") is not None and row.get("source_file") is not None and row.get("source_row_id") is not None:
        return f"{row['case_study']}|{row['source_file']}|{row['source_row_id']}"
    return str(row.get("source_row_id", ""))


def top_numeric_items(value: Any, limit: int = 5) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    numeric: list[tuple[str, float]] = []
    for key, raw in value.items():
        try:
            numeric.append((str(key), float(raw)))
        except (TypeError, ValueError):
            continue
    if not numeric:
        return None
    return dict(sorted(numeric, key=lambda item: item[1], reverse=True)[:limit])


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    btmm_scores = record.get("btmm_scores")
    compact_btmm = None
    if isinstance(btmm_scores, dict):
        compact_btmm = {
            "selected_move_gamma": btmm_scores.get("selected_move_gamma"),
            "sum_gamma": btmm_scores.get("sum_gamma"),
            "top_gamma_moves": btmm_scores.get("top_gamma_moves"),
        }
    return {
        "case_study": record.get("case_study"),
        "domain": record.get("domain"),
        "state_descriptor": record.get("state_descriptor"),
        "selected_action": record.get("selected_action"),
        "alternative_action": record.get("alternative_action"),
        "selection_mechanism": record.get("selection_mechanism"),
        "selected_probability": record.get("selected_probability"),
        "top_candidate_probabilities": top_numeric_items(record.get("candidate_probabilities")),
        "positive_contributors": record.get("positive_contributors"),
        "negative_contributors": record.get("negative_contributors"),
        "decomposed_q_values": record.get("decomposed_q_values"),
        "btmm_scores": compact_btmm,
        "total_margin": record.get("total_margin"),
        "stochasticity_note": record.get("stochasticity_note"),
        "source_file": record.get("source_file"),
        "source_row_id": record.get("source_row_id"),
    }


def ollama_generate(
    base_url: str,
    model: str,
    record: dict[str, Any],
    num_predict: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    timeout: int,
) -> tuple[dict[str, Any] | str, dict[str, Any]]:
    prompt_record = compact_record(record)
    record_json = json.dumps(prompt_record, ensure_ascii=False, sort_keys=True)
    payload: dict[str, Any] = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": USER_PROMPT_TEMPLATE.format(record_json=record_json),
        "format": "json",
        "stream": False,
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
            "top_p": top_p,
        },
    }
    if seed is not None:
        payload["options"]["seed"] = seed

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    raw_text = response_payload.get("response", "")
    try:
        parsed_output: dict[str, Any] | str = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed_output = raw_text

    generation_info = {
        "elapsed_ms": elapsed_ms,
        "eval_count": response_payload.get("eval_count"),
        "eval_duration": response_payload.get("eval_duration"),
        "prompt_eval_count": response_payload.get("prompt_eval_count"),
        "prompt_eval_duration": response_payload.get("prompt_eval_duration"),
        "total_duration": response_payload.get("total_duration"),
        "done_reason": response_payload.get("done_reason"),
    }
    return parsed_output, generation_info


def model_digest(base_url: str, model: str, timeout: int) -> str | None:
    try:
        request = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    for item in payload.get("models", []):
        if item.get("name") == model or item.get("model") == model:
            return item.get("digest")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM verbalizations from XRL explanation records using Ollama.")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--num-predict", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip records already present in the output JSONL.")
    args = parser.parse_args()

    records = load_jsonl(args.records)
    if args.limit is not None:
        records = records[: args.limit]
    existing_keys = load_existing_keys(args.output) if args.resume else set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = model_digest(args.base_url, args.model, args.timeout)

    total = len(records)
    written = 0
    skipped = 0
    failed = 0
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8", newline="\n") as f:
        for index, record in enumerate(records, start=1):
            key = output_key(record)
            if key in existing_keys:
                skipped += 1
                continue
            metadata = {
                "provider": "ollama",
                "model": args.model,
                "model_digest": digest,
                "interface": "Ollama local REST API",
                "generation_date": datetime.now(timezone.utc).isoformat(),
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
                "num_predict": args.num_predict,
                "prompt_version": PROMPT_VERSION,
                "prompt_record_mode": "compact",
            }
            output_row: dict[str, Any] = {
                "case_study": record.get("case_study"),
                "source_file": record.get("source_file"),
                "source_sheet": record.get("source_sheet"),
                "source_row_id": record.get("source_row_id"),
                "llm_metadata": metadata,
                "llm_output": None,
                "generation_info": None,
                "generation_error": None,
            }
            try:
                llm_output, generation_info = ollama_generate(
                    args.base_url,
                    args.model,
                    record,
                    args.num_predict,
                    args.temperature,
                    args.top_p,
                    args.seed,
                    args.timeout,
                )
                output_row["llm_output"] = llm_output
                output_row["generation_info"] = generation_info
                written += 1
            except Exception as exc:  # Keep batch progress inspectable.
                output_row["generation_error"] = f"{type(exc).__name__}: {exc}"
                failed += 1
            f.write(json.dumps(output_row, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            if index == 1 or index % 25 == 0 or index == total:
                print(f"processed={index}/{total} written={written} skipped={skipped} failed={failed}", flush=True)

    summary = {"records": total, "written": written, "skipped": skipped, "failed": failed, "output": str(args.output)}
    print(json.dumps(summary, ensure_ascii=False))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
