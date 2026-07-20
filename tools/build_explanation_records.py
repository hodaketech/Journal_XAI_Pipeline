from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "revision_outputs" / "explanation_records.jsonl"

ACTION_MAP = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
COMPONENT_MEANINGS = {
    "turn": "movement/turning efficiency",
    "goal": "goal progress",
    "blocked": "blocked-position avoidance",
    "safe": "safety",
}
DELTA_FIELDS = {
    "turn": "delta_turn",
    "goal": "delta_goal",
    "blocked": "delta_blocked",
    "safe": "delta_safe",
}


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def parse_literal(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except TypeError:
        pass
    if isinstance(value, (list, tuple, dict)):
        return value
    try:
        return ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return fallback


def parse_list(value: Any) -> list[str]:
    parsed = parse_literal(value, [])
    if isinstance(parsed, (list, tuple, set)):
        return [str(item) for item in parsed]
    return []


def parse_probability_dict(value: Any) -> dict[str, float] | None:
    parsed = parse_literal(value, None)
    if not isinstance(parsed, dict):
        return None
    result: dict[str, float] = {}
    for key, raw in parsed.items():
        number = finite_or_none(raw)
        if number is not None:
            result[str(key)] = number
    return result


def parse_pipe_list(value: Any, item_type: type) -> list[Any]:
    if value is None:
        return []
    parts = [part for part in str(value).split("|") if part != ""]
    out = []
    for part in parts:
        try:
            out.append(item_type(part))
        except ValueError:
            continue
    return out


def action_name(value: Any) -> str:
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        return str(value)
    return ACTION_MAP.get(as_int, str(as_int))


def row_value(row: pd.Series, column: str) -> Any:
    return row[column] if column in row.index else None


def component_margins(row: pd.Series) -> dict[str, float | None]:
    values = {}
    for comp, canonical in DELTA_FIELDS.items():
        raw = row_value(row, f"Δ_{comp}")
        values[canonical] = finite_or_none(raw)
    return values


def contributors(row: pd.Series, source_column: str, role: str) -> list[dict[str, Any]]:
    comps = parse_list(row_value(row, source_column))
    margins = component_margins(row)
    result = []
    for comp in comps:
        result.append(
            {
                "name": comp,
                "value": margins.get(DELTA_FIELDS.get(comp, "")),
                "role": role,
                "meaning": COMPONENT_MEANINGS.get(comp),
                "source_field": source_column,
            }
        )
    return result


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        if hasattr(value, "item"):
            return convert(value.item())
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value

    return convert(record)


def build_case1_records(limit: int | None = None) -> list[dict[str, Any]]:
    dataset = ROOT / "Maze_drQ_Epsilon_Greedy" / "excel-results" / "dataset-final.xlsx"
    validation = ROOT / "Maze_drQ_Epsilon_Greedy" / "excel-results" / "validation-report-final.xlsx"
    df = pd.read_excel(dataset, sheet_name="Episode Log")
    val_df = pd.read_excel(validation, sheet_name="Validation_Report") if validation.exists() else pd.DataFrame()
    if limit is not None:
        df = df.head(limit)

    records = []
    for idx, row in df.iterrows():
        template = None
        if not val_df.empty and idx < len(val_df) and "Explanation" in val_df.columns:
            template = str(val_df.iloc[idx]["Explanation"])
        records.append(
            {
                "case_study": "Case Study 1",
                "domain": "Maze",
                "state_descriptor": {
                    "coordinates": str(row_value(row, "State")),
                    "next_state": str(row_value(row, "Next State")),
                },
                "selected_action": action_name(row_value(row, "Chosen Action")),
                "alternative_action": action_name(row_value(row, "Alternative Action")),
                "selection_mechanism": "epsilon-greedy",
                "selected_probability": None,
                "candidate_probabilities": None,
                "positive_contributors": contributors(row, "MSX+", "positive"),
                "negative_contributors": contributors(row, "MSX-", "negative"),
                "decomposed_q_values": component_margins(row),
                "btmm_scores": None,
                "gamma_values": None,
                "total_margin": finite_or_none(row_value(row, "Total Δ")),
                "stochasticity_note": "epsilon-greedy may select exploratory actions, but this output row does not store an exploration flag.",
                "template_explanation": template,
                "llm_metadata": None,
                "source_file": "Maze_drQ_Epsilon_Greedy/excel-results/dataset-final.xlsx",
                "source_sheet": "Episode Log",
                "source_row_id": int(idx) + 2,
            }
        )
    return [clean_record(record) for record in records]


def build_case2_records(limit: int | None = None) -> list[dict[str, Any]]:
    dataset = ROOT / "Maze_drQ_RWS" / "maze-XRL" / "excel-results" / "evaluation-report-final.xlsx"
    validation = ROOT / "Maze_drQ_RWS" / "maze-XRL" / "excel-results" / "validation-report-final.xlsx"
    df = pd.read_excel(dataset, sheet_name="Episode Log")
    val_df = pd.read_excel(validation, sheet_name="Validation_Report") if validation.exists() else pd.DataFrame()
    if limit is not None:
        df = df.head(limit)

    records = []
    for idx, row in df.iterrows():
        candidate_probs = parse_probability_dict(row_value(row, "total_P"))
        selected = action_name(row_value(row, "Chosen Action"))
        selected_prob = candidate_probs.get(selected) if candidate_probs else None
        template = None
        if not val_df.empty and idx < len(val_df) and "Explanation" in val_df.columns:
            template = str(val_df.iloc[idx]["Explanation"])
        records.append(
            {
                "case_study": "Case Study 2",
                "domain": "Maze",
                "state_descriptor": {
                    "coordinates": str(row_value(row, "State")),
                    "next_state": str(row_value(row, "Next State")),
                },
                "selected_action": selected,
                "alternative_action": action_name(row_value(row, "Alternative Action")),
                "selection_mechanism": "RWS",
                "selected_probability": selected_prob,
                "candidate_probabilities": candidate_probs,
                "positive_contributors": contributors(row, "MSX+", "positive"),
                "negative_contributors": contributors(row, "MSX-", "negative"),
                "decomposed_q_values": {
                    **component_margins(row),
                    "rws_component_probabilities": {
                        "turn": parse_probability_dict(row_value(row, "turn_P")),
                        "goal": parse_probability_dict(row_value(row, "goal_P")),
                        "blocked": parse_probability_dict(row_value(row, "blocked_P")),
                        "safe": parse_probability_dict(row_value(row, "safe_P")),
                    },
                },
                "btmm_scores": None,
                "gamma_values": None,
                "total_margin": finite_or_none(row_value(row, "Total Δ")),
                "stochasticity_note": "RWS samples from normalized action probabilities; the selected action is not necessarily a deterministic argmax.",
                "template_explanation": template,
                "llm_metadata": None,
                "source_file": "Maze_drQ_RWS/maze-XRL/excel-results/evaluation-report-final.xlsx",
                "source_sheet": "Episode Log",
                "source_row_id": int(idx) + 2,
            }
        )
    return [clean_record(record) for record in records]


def infer_selected_move(explanation: Any, moves: list[int], gammas: list[float]) -> tuple[int | None, float | None]:
    text = "" if explanation is None or pd.isna(explanation) else str(explanation)
    match = re.search(r"(?:move|nước)\s+(\d+)", text, flags=re.IGNORECASE)
    if not match:
        return None, None
    move = int(match.group(1))
    if move not in moves:
        return move, None
    idx = moves.index(move)
    return move, gammas[idx] if idx < len(gammas) else None


def build_case3_records(limit: int | None = None) -> list[dict[str, Any]]:
    csv_path = ROOT / "Connect6" / "BitBoard" / "evaluation-validation-dataset-processed.csv"
    validation = ROOT / "Connect6" / "BitBoard" / "logs" / "validation_result.xlsx"
    df = pd.read_csv(csv_path)
    val_df = pd.read_excel(validation, sheet_name=0) if validation.exists() else pd.DataFrame()
    if limit is not None:
        df = df.head(limit)

    records = []
    for idx, row in df.iterrows():
        moves = parse_pipe_list(row_value(row, "moves"), int)
        gammas = parse_pipe_list(row_value(row, "gamma_values"), float)
        probability_map = {str(move): gammas[pos] for pos, move in enumerate(moves) if pos < len(gammas)}
        explanation = None
        if not val_df.empty and idx < len(val_df) and "human_explanation" in val_df.columns:
            explanation = val_df.iloc[idx]["human_explanation"]

        selected_move, selected_gamma = infer_selected_move(explanation, moves, gammas)
        if selected_move is None and gammas:
            best_idx = int(max(range(len(gammas)), key=lambda pos: gammas[pos]))
            selected_move = moves[best_idx] if best_idx < len(moves) else best_idx
            selected_gamma = gammas[best_idx]

        top_pairs = sorted(probability_map.items(), key=lambda item: item[1], reverse=True)[:5]
        top_gamma_moves = {move: gamma for move, gamma in top_pairs}
        low_prob_note = ""
        if selected_gamma is not None and selected_gamma < 0.05:
            low_prob_note = " The selected move has low probability and should be described as stochastic rather than deterministic."

        records.append(
            {
                "case_study": "Case Study 3",
                "domain": "Connect6",
                "state_descriptor": {
                    "game_id": int(row_value(row, "game_id")),
                    "move_id": int(row_value(row, "move_id")),
                    "iteration": int(row_value(row, "iteration")),
                    "player": int(row_value(row, "player")),
                    "num_candidate_moves": int(row_value(row, "num_moves")),
                },
                "selected_action": selected_move,
                "alternative_action": None,
                "selection_mechanism": "BTMM+RWS",
                "selected_probability": finite_or_none(selected_gamma),
                "candidate_probabilities": probability_map,
                "positive_contributors": [],
                "negative_contributors": [],
                "decomposed_q_values": None,
                "btmm_scores": {
                    "top_gamma_moves": top_gamma_moves,
                    "selected_move_gamma": finite_or_none(selected_gamma),
                    "sum_gamma": finite_or_none(row_value(row, "sum_gamma")),
                },
                "gamma_values": probability_map,
                "total_margin": None,
                "stochasticity_note": "BTMM+RWS samples moves from normalized gamma values." + low_prob_note,
                "template_explanation": None if explanation is None or pd.isna(explanation) else str(explanation),
                "llm_metadata": None,
                "source_file": "Connect6/BitBoard/evaluation-validation-dataset-processed.csv",
                "source_sheet": None,
                "source_row_id": int(idx) + 2,
            }
        )
    return [clean_record(record) for record in records]


def write_jsonl(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized XRL explanation records from existing outputs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument("--limit-per-case", type=int, default=None, help="Optional maximum rows per case.")
    args = parser.parse_args()

    records = []
    records.extend(build_case1_records(args.limit_per_case))
    records.extend(build_case2_records(args.limit_per_case))
    records.extend(build_case3_records(args.limit_per_case))
    write_jsonl(records, args.output)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
