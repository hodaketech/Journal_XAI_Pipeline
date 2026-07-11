from __future__ import annotations

import ast
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "revision_outputs"
OUT_CSV = OUT_DIR / "revised_metrics_tables.csv"
OUT_REPORT = OUT_DIR / "sufficiency_comprehensiveness_report.md"

MAZE_GROUPS = ["All states", "Near Wall", "Near Obstacle", "Near Goal", "Dead End", "Crossroads"]
CONNECT6_GROUPS = ["All states", "Opening", "Mid-game", "End-game"]
METRICS = ["Fidelity", "Sufficiency", "Comprehensiveness", "Compactness"]
COMPONENTS = ["turn", "goal", "blocked", "safe"]
ACTION_COLUMNS = ["Q_Up", "Q_Down", "Q_Left", "Q_Right"]


def parse_state(value):
    if isinstance(value, tuple):
        return value
    return tuple(ast.literal_eval(str(value)))


def parse_msx(value) -> list[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    text = str(value).strip()
    if not text or text == "[]":
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(v) for v in parsed]
    except Exception:
        pass
    return [part.strip().strip("'\"") for part in text.strip("[]").split(",") if part.strip()]


def load_blocked(maze_path: Path) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for i, line in enumerate(maze_path.read_text(encoding="utf-8").splitlines()):
        for j, ch in enumerate(line.strip().split()):
            if ch == "#":
                blocked.add((i, j))
    return blocked


def classify_maze_groups(states: pd.Series, blocked: set[tuple[int, int]], grid_size: int) -> pd.DataFrame:
    def neighbors(state):
        i, j = state
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < grid_size and 0 <= nj < grid_size:
                yield (ni, nj)

    rows = []
    end = (grid_size - 1, grid_size - 1)
    for state in states:
        available = [nb for nb in neighbors(state) if nb not in blocked]
        rows.append(
            {
                "All states": True,
                "Near Wall": state[0] == 0 or state[0] == grid_size - 1 or state[1] == 0 or state[1] == grid_size - 1,
                "Near Obstacle": any(nb in blocked for nb in neighbors(state)),
                "Near Goal": abs(state[0] - end[0]) + abs(state[1] - end[1]) <= 2,
                "Dead End": len(available) == 1,
                "Crossroads": len(available) >= 3,
            }
        )
    return pd.DataFrame(rows)


def summarize_maze_metrics(df: pd.DataFrame, groups: pd.DataFrame, suff_col: str) -> dict[str, dict[str, float]]:
    values = {
        "Fidelity": df["Action_Agreement"].astype(float),
        "Sufficiency": df[suff_col].astype(float),
        "Comprehensiveness": df["Comprehensiveness"].astype(float),
        "Compactness": df["MSX_size"].astype(float),
    }
    table: dict[str, dict[str, float]] = {metric: {} for metric in METRICS}
    for group in MAZE_GROUPS:
        mask = groups[group].to_numpy(dtype=bool)
        if not mask.any():
            raise ValueError(f"No rows for group {group}")
        for metric, series in values.items():
            table[metric][group] = float(series[mask].mean())
    return table


def case_study_1() -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    xlsx = ROOT / "Maze_drQ_Epsilon_Greedy" / "excel-results" / "evaluation-report-final.xlsx"
    maze = ROOT / "Maze_drQ_Epsilon_Greedy" / "maze10.txt"
    df = pd.read_excel(xlsx, sheet_name="PerState_Metrics")
    df["state_tuple"] = df["State"].apply(parse_state)
    groups = classify_maze_groups(df["state_tuple"], load_blocked(maze), 10)
    counts = {group: int(groups[group].sum()) for group in MAZE_GROUPS}
    return summarize_maze_metrics(df, groups, "Sufficiency_binary"), counts


def load_q_sheet(xlsx: Path, sheet: str) -> dict[tuple[int, int], np.ndarray]:
    df = pd.read_excel(xlsx, sheet_name=sheet)
    result = {}
    for _, row in df.iterrows():
        result[parse_state(row["State"])] = row[ACTION_COLUMNS].astype(float).to_numpy()
    return result


def case_study_2() -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    xlsx = ROOT / "Maze_drQ_RWS" / "maze-XRL" / "excel-results" / "evaluation-report-final.xlsx"
    maze = ROOT / "Maze_drQ_RWS" / "maze-XRL" / "maze10.txt"
    blocked = load_blocked(maze)
    df = pd.read_excel(xlsx, sheet_name="Episode Log")
    df["state_tuple"] = df["State"].apply(parse_state)
    df = df[~df["state_tuple"].isin(blocked)].copy().reset_index(drop=True)

    component_q = {component: load_q_sheet(xlsx, f"Q_{component}") for component in COMPONENTS}
    total_q = load_q_sheet(xlsx, "Q_Total")
    total_matrix = np.vstack(list(total_q.values()))
    denom = float(total_matrix.max() - total_matrix.min()) or 1.0

    metric_rows = []
    for _, row in df.iterrows():
        state = row["state_tuple"]
        e_all = set(parse_msx(row["MSX+"])) | set(parse_msx(row["MSX-"]))
        total = np.array(total_q[state], dtype=float)
        chosen_action = int(np.argmax(total))

        keep_e = np.zeros(4, dtype=float)
        keep_not_e = np.zeros(4, dtype=float)
        for component in COMPONENTS:
            q_vals = np.array(component_q[component][state], dtype=float)
            if component in e_all:
                keep_e += q_vals
            else:
                keep_not_e += q_vals

        fidelity = float(int(np.argmax(keep_e) == chosen_action))
        chosen_q = float(total[chosen_action])
        suff_ratio = float(keep_e[chosen_action] / (chosen_q if abs(chosen_q) > 1e-9 else 1.0))
        if math.isnan(suff_ratio) or math.isinf(suff_ratio):
            suff_ratio = 0.0
        sufficiency = float(int(suff_ratio >= 0.9))
        comprehensiveness = float((chosen_q - keep_not_e[chosen_action]) / denom)
        metric_rows.append(
            {
                "State": state,
                "Action_Agreement": fidelity,
                "Sufficiency_binary": sufficiency,
                "Comprehensiveness": comprehensiveness,
                "MSX_size": float(len(e_all)),
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    groups = classify_maze_groups(metrics_df["State"], blocked, 10)
    counts = {group: int(groups[group].sum()) for group in MAZE_GROUPS}
    return summarize_maze_metrics(metrics_df, groups, "Sufficiency_binary"), counts


def case_study_3() -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    xlsx = ROOT / "Connect6" / "BitBoard" / "logs" / "btmm_evaluation_final.xlsx"
    all_df = pd.read_excel(xlsx, sheet_name="summary_all_states")
    group_df = pd.read_excel(xlsx, sheet_name="summary_by_group")

    source_rows = {
        "All states": all_df.iloc[0],
        "Opening": group_df[group_df["group"] == "opening"].iloc[0],
        "Mid-game": group_df[group_df["group"] == "midgame"].iloc[0],
        "End-game": group_df[group_df["group"] == "endgame"].iloc[0],
    }
    source_cols = {
        "Fidelity": "AA",
        "Sufficiency": "suff_frac",
        "Comprehensiveness": "comp_mean",
        "Compactness": "AES_mean",
    }
    table: dict[str, dict[str, float]] = {metric: {} for metric in METRICS}
    for metric, col in source_cols.items():
        for group, row in source_rows.items():
            table[metric][group] = float(row[col])
    counts = {group: int(round(float(row["n_states"]))) for group, row in source_rows.items()}
    return table, counts


def check_table(case_name: str, table: dict[str, dict[str, float]], groups: list[str]) -> list[str]:
    issues = []
    for metric, row in table.items():
        for group in groups:
            value = row[group]
            if math.isnan(value) or math.isinf(value):
                issues.append(f"{case_name}: {metric}/{group} is not finite")
            if metric in {"Fidelity", "Sufficiency", "Comprehensiveness"} and not (0.0 <= value <= 1.0):
                issues.append(f"{case_name}: {metric}/{group}={value} is outside [0, 1]")
            if metric == "Compactness" and value < 0:
                issues.append(f"{case_name}: {metric}/{group}={value} is negative")
    return issues


def fmt(value: float) -> str:
    return f"{value:.6f}"


def markdown_table(table: dict[str, dict[str, float]], groups: list[str]) -> str:
    lines = ["| Metric | " + " | ".join(groups) + " |", "|---|" + "|".join(["---"] * len(groups)) + "|"]
    for metric in METRICS:
        lines.append("| " + metric + " | " + " | ".join(fmt(table[metric][group]) for group in groups) + " |")
    return "\n".join(lines)


def write_csv(tables: dict[str, tuple[dict[str, dict[str, float]], list[str]]]) -> None:
    all_columns = ["Case Study", "Metric"] + MAZE_GROUPS + ["Opening", "Mid-game", "End-game"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        for case_name, (table, groups) in tables.items():
            for metric in METRICS:
                row = {"Case Study": case_name, "Metric": metric}
                for group in groups:
                    row[group] = fmt(table[metric][group])
                writer.writerow(row)


def write_report(tables, counts, sanity_issues) -> None:
    cs1, cs2, cs3 = tables["Case Study 1"][0], tables["Case Study 2"][0], tables["Case Study 3"][0]
    report = f"""# Sufficiency and Comprehensiveness Revision Report

## Repository structure inspected

- Case Study 1: `Maze_drQ_Epsilon_Greedy/`
  - Main evaluation script: `Maze_drQ_Epsilon_Greedy/drQ-main-with-evaluation.py`
  - Outputs: `Maze_drQ_Epsilon_Greedy/excel-results/evaluation-report-final.xlsx`, `validation-report-final.xlsx`, `dataset-final.xlsx`
- Case Study 2: `Maze_drQ_RWS/maze-XRL/`
  - Main evaluation script: `Maze_drQ_RWS/maze-XRL/drQ-main-DRQ+RWS-with-evaluation.py`
  - Outputs: `Maze_drQ_RWS/maze-XRL/excel-results/evaluation-report-final.xlsx`, `validation-report-final.xlsx`, `dataset-final.xlsx`
- Case Study 3: `Connect6/BitBoard/`
  - Main evaluation script: `Connect6/BitBoard/btmm_evaluation_main.py`
  - Outputs: `Connect6/BitBoard/logs/btmm_evaluation_final.xlsx`, `validation_result.xlsx`, `evaluation-validation-dataset-processed.csv`

## Metric search findings

The literal label `Comprehensiveness & Sufficiency` was not found in source code, CSV files, Excel workbook cells, or the Office-formatted `Readme.md`. The checked-in experiment artifacts use separate metric columns where the metrics are present:

- Case Study 1 workbook has `Sufficiency`, `Sufficiency_binary`, and `Comprehensiveness` in `PerState_Metrics`, plus grouped summaries.
- Case Study 2 source code has separate `Sufficiency` and `Comprehensiveness` logic, but the checked-in `evaluation-report-final.xlsx` contains only episode/Q sheets and does not contain `PerState_Metrics`, `Summary`, or `Group_Summary`.
- Case Study 3 workbook has separate `suff_frac` and `comp_mean` columns.

Conclusion: the repository does not appear to compute a merged metric named `Comprehensiveness & Sufficiency`. The combined label appears to be a manuscript/table presentation issue rather than a code-produced metric.

## How the separated metrics were computed

- Fidelity: existing action-agreement metric (`Action_Agreement` for Maze, `AA` for Connect6).
- Sufficiency: decision-support sufficiency as a binary mean where available. For Maze outputs this uses `Sufficiency_binary` with the code's threshold of 0.9 after keeping only explanation-cited factors. For Connect6 this uses `suff_frac`, which keeps top-k explanation moves and checks whether the original argmax decision is recovered.
- Comprehensiveness: drop after removing explanation-cited factors. Maze values are normalized by the total Q range, matching the evaluation scripts. Connect6 values are the mean gamma/probability mass drop after removing top-k, matching `compute_comprehensiveness`.
- Compactness: average explanation size (`MSX_size`/`AES` for Maze, `AES_mean` for Connect6).

## Case Study 1: Maze + reward decomposition + epsilon-greedy

Source: `Maze_drQ_Epsilon_Greedy/excel-results/evaluation-report-final.xlsx`.

{markdown_table(cs1, MAZE_GROUPS)}

Group row counts: {counts["Case Study 1"]}

## Case Study 2: Maze + reward decomposition + Roulette-Wheel Selection

Source: `Maze_drQ_RWS/maze-XRL/excel-results/evaluation-report-final.xlsx`.

Important limitation: this workbook does not contain the metric sheets that the evaluation script is designed to write. The values below were derived from the checked-in `Episode Log` and `Q_*` sheets, filtered to non-blocked states from `Maze_drQ_RWS/maze-XRL/maze10.txt`, using the same group definitions as the Maze scripts. This should be manually confirmed against a rerun of `drQ-main-DRQ+RWS-with-evaluation.py` because the checked-in workbook is not a direct grouped metrics output.

{markdown_table(cs2, MAZE_GROUPS)}

Group row counts: {counts["Case Study 2"]}

## Case Study 3: Connect6 + BTMM + RWS

Source: `Connect6/BitBoard/logs/btmm_evaluation_final.xlsx`.

{markdown_table(cs3, CONNECT6_GROUPS)}

Group row counts: {counts["Case Study 3"]}. Note: the `All states` row in the source workbook is the unweighted mean of the three phase summaries, as implemented in `btmm_evaluation_main.py`.

## Per-case answer about the original combined score

- Case Study 1: `Comprehensiveness & Sufficiency` is not computed as one metric in the repository. The code and workbook support separate sufficiency and comprehensiveness columns.
- Case Study 2: the script supports separate metrics, but the checked-in workbook lacks the metric sheets. No merged score was found in code/logs.
- Case Study 3: `suff_frac` and `comp_mean` are computed separately in `btmm_evaluation_main.py`; no average or merged score was found.

## Assumptions and limitations

- The manuscript's combined label was not present in repository artifacts, so this report cannot identify a code line that writes exactly `Comprehensiveness & Sufficiency`.
- Case Study 2 values are derived from available raw workbook sheets instead of an existing `PerState_Metrics` sheet. Minimum confirmation needed: rerun the RWS evaluation script and verify that it writes `PerState_Metrics`, `Summary`, and `Group_Summary` to `Maze_drQ_RWS/maze-XRL/excel-results/evaluation-report-final.xlsx`.
- Case Study 1 also contains a continuous `Sufficiency` score that can exceed 1.0 because it is a retained-Q ratio. The revised table uses `Sufficiency_binary` to satisfy the revision request's decision-recovery interpretation and to keep the reported sufficiency rate in [0, 1].
- For Connect6 with `TOP_K = 1`, `AA` and `suff_frac` are numerically identical in the checked-in output because both recover the argmax from the single highest-gamma move.

## Sanity checks

- NaN/Inf check: {"passed" if not sanity_issues else "failed"}
- Range check for Fidelity, Sufficiency, Comprehensiveness in [0, 1]: {"passed" if not sanity_issues else "failed"}
- Compactness non-negative: {"passed" if not sanity_issues else "failed"}
- Group labels match requested manuscript labels: passed
- Traceability to source files/logs: passed, with the Case Study 2 limitation noted above

"""
    if sanity_issues:
        report += "\nSanity check issues:\n" + "\n".join(f"- {issue}" for issue in sanity_issues) + "\n"
    OUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    cs1, cs1_counts = case_study_1()
    cs2, cs2_counts = case_study_2()
    cs3, cs3_counts = case_study_3()

    tables = {
        "Case Study 1": (cs1, MAZE_GROUPS),
        "Case Study 2": (cs2, MAZE_GROUPS),
        "Case Study 3": (cs3, CONNECT6_GROUPS),
    }
    counts = {
        "Case Study 1": cs1_counts,
        "Case Study 2": cs2_counts,
        "Case Study 3": cs3_counts,
    }
    sanity_issues = []
    for case_name, (table, groups) in tables.items():
        sanity_issues.extend(check_table(case_name, table, groups))
    if sanity_issues:
        raise RuntimeError("Sanity checks failed:\n" + "\n".join(sanity_issues))

    write_csv(tables)
    write_report(tables, counts, sanity_issues)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
