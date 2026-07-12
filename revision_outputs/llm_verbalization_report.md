# LLM / Natural-Language Verbalization Revision Report

## Executive finding

No implemented LLM integration was found in the repository. I found no OpenAI/ChatGPT API calls, no LLM client library use, no model names, no prompt files, no temperature/top-p settings, and no script/notebook that sends explanation evidence to an LLM.

The repository does contain ordinary code templates that generate human-readable explanation text for validation reports. These templates convert numerical evidence such as selected action, alternative action, MSX positive/negative contributors, component deltas, total margin, and gamma probabilities into fixed natural-language strings. This is not an LLM pipeline.

Given the supervisor clarification that ChatGPT was used offline to verbalize numerical evidence, the recommended manuscript strategy is **Strategy B: offline LLM-assisted verbalization**, with a clear statement that the LLM was not part of the RL policy, scoring, selection, training, or metric evaluation code.

## Search evidence

Repository-wide searches were run for LLM and prompt terms, including `LLM`, `large language model`, `ChatGPT`, `GPT`, `OpenAI`, `prompt`, `system prompt`, `custom prompt`, `verbalize`, `verbalization`, `natural language`, `generated explanation`, `template`, `renderer`, `human-readable`, `local explanation`, and `global explanation`.

Findings:

- No LLM API use was found.
- No prompt template file was found.
- No model name, temperature, top-p, seed, or generation date was found.
- The only `ChatGPT` hit was `Connect6/BitBoard/game.py:5`, a comment reading `@bitboard-refactor: ChatGPT July 2025`. This is a code-refactor note, not an explanation generation module.
- Natural-language explanation columns were found in validation workbooks:
  - `Maze_drQ_Epsilon_Greedy/excel-results/validation-report-final.xlsx`, sheet `Validation_Report`, column `Explanation`.
  - `Maze_drQ_RWS/maze-XRL/excel-results/validation-report-final.xlsx`, sheet `Validation_Report`, column `Explanation`.
  - `Connect6/BitBoard/logs/validation_result.xlsx`, sheet `Kết quả kiểm tra`, column `human_explanation`.

## What produces natural-language text in the repository

### Maze case studies

Both Maze auto-validation scripts contain a class named `ExplanationGenerator`:

- `Maze_drQ_Epsilon_Greedy/drQ-main-auto-validation.py:174`
- `Maze_drQ_RWS/maze-XRL/drQ-main-DRQ+RWS-auto-validation.py:174`

The generator builds text by fixed code branches:

- `generate_explanation(...)`: lines 178-197 in both scripts.
- `_create_detailed_explanation(...)`: lines 199-264 in both scripts.
- A comment explicitly says `Build natural language explanation`: line 231 in both scripts.
- The final text is written into the Excel report as `Explanation`: lines 682-704 in both scripts.

This is template/code-based rendering, not LLM generation.

### Connect6

`Connect6/BitBoard/auto-validation.py` generates `human_explanation` using fixed thresholds over RWS gamma values:

- `human_explanation` column initialized: line 15.
- `explain_move(...)` function: line 19.
- The function branches on `move_gamma` thresholds and returns fixed strings: lines 39-67.
- The generated explanation is assigned to `human_explanation`: lines 133-135.
- The report is written to `logs/validation_result.xlsx`: lines 203-206.

This is also template/code-based rendering, not LLM generation.

## Numerical evidence available as LLM input

### Case Study 1: Maze + reward decomposition + epsilon-greedy

Source folders and scripts:

- `Maze_drQ_Epsilon_Greedy/drQ-main-with-evaluation.py`
- `Maze_drQ_Epsilon_Greedy/drQ-main-auto-validation.py`
- `Maze_drQ_Epsilon_Greedy/excel-results/dataset-final.xlsx`
- `Maze_drQ_Epsilon_Greedy/excel-results/evaluation-report-final.xlsx`
- `Maze_drQ_Epsilon_Greedy/excel-results/validation-report-final.xlsx`

Traceable evidence:

- Selected action: `dataset-final.xlsx`, sheet `Episode Log`, column `Chosen Action`; produced in `drQ-main-with-evaluation.py:305`.
- Alternative action: `Episode Log`, column `Alternative Action`; produced at line 306.
- RDX/component margins: `Episode Log`, columns `Δ_turn`, `Δ_goal`, `Δ_blocked`, `Δ_safe`; computed by `reward_difference_explanation(...)` at lines 150-165.
- MSX positive contributors: `Episode Log`, column `MSX+`; produced at line 312.
- MSX negative contributors: `Episode Log`, column `MSX-`; produced at line 313.
- MSX construction: `minimal_sufficient_explanation(...)` at lines 168-213.
- Decomposed Q-values: workbook sheets `Q_turn`, `Q_goal`, `Q_blocked`, `Q_safe`; written at lines 328-342.
- Total Q-values: workbook sheet `Q_Total`; written at lines 344-357.
- Total improvement/score margin: `Episode Log`, column `Total Δ`; written at line 311.
- Template natural-language output: `validation-report-final.xlsx`, sheet `Validation_Report`, column `Explanation`.

### Case Study 2: Maze + reward decomposition + Roulette-Wheel Selection

Source folders and scripts:

- `Maze_drQ_RWS/maze-XRL/drQ-main-DRQ+RWS-with-evaluation.py`
- `Maze_drQ_RWS/maze-XRL/drQ-main-DRQ+RWS-auto-validation.py`
- `Maze_drQ_RWS/maze-XRL/excel-results/evaluation-report-final.xlsx`
- `Maze_drQ_RWS/maze-XRL/excel-results/dataset-final.xlsx`
- `Maze_drQ_RWS/maze-XRL/excel-results/validation-report-final.xlsx`

Traceable evidence:

- Selected action: `evaluation-report-final.xlsx`, sheet `Episode Log`, column `Chosen Action`; produced at `drQ-main-DRQ+RWS-with-evaluation.py:400`.
- Alternative action: `Episode Log`, column `Alternative Action`; produced at line 401.
- RDX/component margins: `Episode Log`, columns `Δ_turn`, `Δ_goal`, `Δ_blocked`, `Δ_safe`; computed by `reward_difference_explanation(...)` at lines 222-236.
- MSX positive contributors and sums: `Episode Log`, columns `MSX+`, `Sum MSX+`; produced at lines 408-409.
- MSX negative contributors and sums: `Episode Log`, columns `MSX-`, `Sum MSX-`; produced at lines 411-412.
- MSX construction: `minimal_sufficient_explanation(...)` at lines 239-287.
- Decomposed Q-values: workbook sheets `Q_turn`, `Q_goal`, `Q_blocked`, `Q_safe`; written at lines 431-445.
- Total Q-values: workbook sheet `Q_Total`; written at lines 460.
- RWS/selection probabilities: `Episode Log` columns `turn_P`, `goal_P`, `blocked_P`, `safe_P`, `total_P`; produced at lines 396 and 416-417. A separate `Roulette_Probs` sheet is written at lines 468-477.
- Total improvement/score margin: `Episode Log`, column `Total Δ`; written at line 406.
- Template natural-language output: `validation-report-final.xlsx`, sheet `Validation_Report`, column `Explanation`.

### Case Study 3: Connect6 + BTMM + RWS

Source folders and scripts:

- `Connect6/BitBoard/evaluation-validation-dataset-processed.csv`
- `Connect6/BitBoard/auto-validation.py`
- `Connect6/BitBoard/btmm_evaluation_main.py`
- `Connect6/BitBoard/train_btmm_update_evaluation.py`
- `Connect6/BitBoard/logs/validation_result.xlsx`
- `Connect6/BitBoard/logs/btmm_evaluation_final.xlsx`

Traceable evidence:

- Candidate moves: `evaluation-validation-dataset-processed.csv`, column `moves`; parsed in `btmm_evaluation_main.py:135`.
- Selection probabilities / BTMM-like gamma values: `evaluation-validation-dataset-processed.csv`, column `gamma_values`; parsed in `btmm_evaluation_main.py:136`.
- Selected index/action by maximum gamma for evaluation: `btmm_evaluation_main.py:143`.
- Sufficiency and comprehensiveness from top-k gamma evidence: `compute_sufficiency(...)` at line 54 and `compute_comprehensiveness(...)` at line 65.
- Aggregated BTMM/RWS evaluation outputs: `logs/btmm_evaluation_final.xlsx`, sheets `summary_all_states` and `summary_by_group`; written at lines 250-252.
- Template RWS human explanation: `auto-validation.py:19-67`, saved to `logs/validation_result.xlsx`.
- Additional training-time evaluator exists in `train_btmm_update_evaluation.py`, including `compute_comprehensiveness_sufficiency(...)` at lines 67-115 and `compute_explanation_metrics(...)` at lines 117-168, but the checked-in final workbook used for the paper-style summary is aggregated.

## Table 3 traceability

The repository does not contain the manuscript Table 3 text. The provided Section 5.4 excerpt mentions Table 3, but the actual Table 3 local/global explanation examples were not present in the repository text or in the user-provided excerpt.

Therefore:

- Exact Table 3 explanations cannot be traced to source rows from the repository alone.
- Maze local examples may be manually matched against `validation-report-final.xlsx` rows if Table 3 text is supplied.
- Connect6 local RWS/gamma examples may be manually matched against `logs/validation_result.xlsx` rows if Table 3 text is supplied.
- Global explanation examples cannot be traced to a specific source file in the current repository unless the supervisor provides the aggregation procedure or source notes used to write them.

Recommended manual confirmation: provide the exact Table 3 text and identify whether each sentence came from a validation workbook row, an offline ChatGPT output, or manual author editing.

## Missing reproducibility information

The following details were **not found in repository**:

- LLM provider.
- Model name/version.
- ChatGPT interface or API use.
- Date or date range of generation.
- Temperature.
- Top-p.
- Seed.
- Prompt text.
- System prompt.
- Input schema supplied to the LLM.
- Output schema required from the LLM.
- Number of examples generated.
- Whether outputs were edited manually.
- Whether outputs were generated in English first, Vietnamese first, or translated.
- Whether the same prompt was used for Maze and Connect6.
- Whether global explanations were generated from aggregate metrics, selected episodes, or author-written summaries.

## Recommended manuscript strategy

Use **Strategy B** if the supervisor confirms offline ChatGPT use:

> We used the RL/XRL pipeline to compute all policy decisions, decomposed value evidence, RDX/MSX contributors, BTMM scores, and RWS probabilities. Natural-language explanations were produced as an offline verbalization step from these numerical records. The LLM was not used for policy learning, action selection, reward decomposition, explanation scoring, or metric evaluation. It was used only to convert structured evidence records into concise human-readable text following a fixed schema: selected action, compared alternative, top positive contributors, top negative contributors when present, and a short takeaway.

If model and prompt details cannot be recovered, do not overclaim reproducibility. Add:

> The original offline ChatGPT verbalization settings were not logged in the source repository. To avoid conflating this verbalization aid with the implemented RL/XRL algorithms, we report the numerical evidence and structured fields used for each explanation and clarify that the LLM output is illustrative rather than an executable component of the released code.

If the supervisor cannot reliably confirm ChatGPT details, use **Strategy C**:

> Explanations are represented as a structured explanation schema derived from numerical evidence. Human-readable examples in the paper are rendered summaries of this schema and should not be described as part of the implemented RL system.

## Suggested replacement for Section 5.4

Replace the sentence:

> We therefore adopt a two-stage normalization procedure that makes explanations structured and readily renderable by large language models (LLMs).

with:

> We therefore adopt a two-stage normalization procedure that converts numerical explanation evidence into a structured record. This record can be rendered directly by deterministic templates or, in the manuscript examples, verbalized offline by ChatGPT for readability.

Replace:

> This normalization turns raw numeric signals into legible tuples that LLMs can verbalize consistently.

with:

> This normalization turns raw numeric signals into legible tuples that separate the implemented explanation evidence from any downstream natural-language rendering.

Replace:

> The Table 3 illustrates the sample generated explanation by LLM system for both Maze and Connect6 problem in Local and Global context.

with:

> Table 3 illustrates human-readable verbalizations of the structured explanation records for Maze and Connect6. These verbalizations were prepared offline from the numerical evidence and were not used by the RL agents during training, action selection, explanation scoring, or evaluation.

Optional stronger wording if the supervisor supplies model/prompt details:

> Table 3 illustrates offline LLM-assisted verbalizations of the structured explanation records. The LLM input consisted only of the selected action, alternative action, top positive and negative contributors, component margins or BTMM/RWS probabilities, and a requested fixed output schema. The LLM was used only for presentation.

## Suggested response to reviewer

> We thank the reviewer for pointing out that the role of the LLM was underspecified. We have revised the manuscript to clarify that no LLM is used inside the reinforcement-learning implementation, policy, scoring, selection, or quantitative evaluation pipeline. The implemented system produces structured numerical explanation records, including decomposed Q-value margins, RDX/MSX contributors, BTMM/RWS scores, and selection probabilities. ChatGPT was used only offline as a verbalization aid to convert these structured records into concise natural-language examples for human readers. We have added a description of this offline verbalization protocol and separated it from the algorithmic and experimental components. Where exact generation settings were not logged in the repository, we now state this limitation and provide the structured input fields needed to reproduce or replace the verbalization step.

If prompt/model details are provided, append:

> The revision now reports the model/version, generation date, prompt template, decoding settings, input schema, and output schema.

If prompt/model details cannot be provided, append:

> Because the original ChatGPT session metadata was not retained, we avoid claiming that the LLM-generated prose is exactly reproducible; instead, we make the numerical evidence and schema reproducible and treat the prose as illustrative.

## Information to request from supervisor

Ask the supervisor for:

1. Exact ChatGPT model or interface used, e.g. ChatGPT web app model name/version or OpenAI API model.
2. Approximate generation date or date range.
3. Whether temperature/top-p/seed were controlled. If ChatGPT web UI was used and these were unavailable, state that.
4. Full custom prompt or best available reconstruction.
5. Whether a system prompt was used.
6. Input records supplied to ChatGPT for Table 3 examples.
7. Exact Table 3 local and global explanation texts.
8. Whether outputs were edited by authors after generation.
9. Whether the same prompt was used for Maze and Connect6.
10. Whether global explanations were based on aggregate metrics, selected trajectories, or author summaries.

## Bottom line

The repository supports this corrected claim:

> The code computes numerical explanation evidence and, in validation scripts, can render template-based natural-language explanations. Any ChatGPT/LLM use was offline prose verbalization for the manuscript and is not part of the released RL implementation or evaluation pipeline.
