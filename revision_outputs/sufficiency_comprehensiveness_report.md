# Sufficiency and Comprehensiveness Revision Report

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

| Metric | All states | Near Wall | Near Obstacle | Near Goal | Dead End | Crossroads |
|---|---|---|---|---|---|---|
| Fidelity | 0.785714 | 0.812500 | 0.781818 | 1.000000 | 1.000000 | 0.909091 |
| Sufficiency | 0.589286 | 0.687500 | 0.600000 | 0.750000 | 0.666667 | 0.818182 |
| Comprehensiveness | 0.500705 | 0.475906 | 0.509808 | 0.734686 | 0.484980 | 0.602176 |
| Compactness | 1.589286 | 1.750000 | 1.618182 | 1.500000 | 1.777778 | 1.636364 |

Group row counts: {'All states': 56, 'Near Wall': 16, 'Near Obstacle': 55, 'Near Goal': 4, 'Dead End': 9, 'Crossroads': 11}

## Case Study 2: Maze + reward decomposition + Roulette-Wheel Selection

Source: `Maze_drQ_RWS/maze-XRL/excel-results/evaluation-report-final.xlsx`.

Important limitation: this workbook does not contain the metric sheets that the evaluation script is designed to write. The values below were derived from the checked-in `Episode Log` and `Q_*` sheets, filtered to non-blocked states from `Maze_drQ_RWS/maze-XRL/maze10.txt`, using the same group definitions as the Maze scripts. This should be manually confirmed against a rerun of `drQ-main-DRQ+RWS-with-evaluation.py` because the checked-in workbook is not a direct grouped metrics output.

| Metric | All states | Near Wall | Near Obstacle | Near Goal | Dead End | Crossroads |
|---|---|---|---|---|---|---|
| Fidelity | 0.750000 | 0.875000 | 0.745455 | 0.750000 | 0.888889 | 0.636364 |
| Sufficiency | 0.589286 | 0.687500 | 0.600000 | 0.750000 | 0.333333 | 0.818182 |
| Comprehensiveness | 0.474835 | 0.463714 | 0.483468 | 0.722373 | 0.355980 | 0.594962 |
| Compactness | 1.732143 | 1.687500 | 1.763636 | 1.250000 | 1.666667 | 1.818182 |

Group row counts: {'All states': 56, 'Near Wall': 16, 'Near Obstacle': 55, 'Near Goal': 4, 'Dead End': 9, 'Crossroads': 11}

## Case Study 3: Connect6 + BTMM + RWS

Source: `Connect6/BitBoard/logs/btmm_evaluation_final.xlsx`.

| Metric | All states | Opening | Mid-game | End-game |
|---|---|---|---|---|
| Fidelity | 0.860229 | 0.812030 | 0.865672 | 0.902985 |
| Sufficiency | 0.860229 | 0.812030 | 0.865672 | 0.902985 |
| Comprehensiveness | 0.106553 | 0.030232 | 0.059366 | 0.230062 |
| Compactness | 46.360341 | 68.857143 | 50.052239 | 20.171642 |

Group row counts: {'All states': 134, 'Opening': 133, 'Mid-game': 134, 'End-game': 134}. Note: the `All states` row in the source workbook is the unweighted mean of the three phase summaries, as implemented in `btmm_evaluation_main.py`.

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

- NaN/Inf check: passed
- Range check for Fidelity, Sufficiency, Comprehensiveness in [0, 1]: passed
- Compactness non-negative: passed
- Group labels match requested manuscript labels: passed
- Traceability to source files/logs: passed, with the Case Study 2 limitation noted above

