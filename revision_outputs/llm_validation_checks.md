# Validation Checks for LLM-Verbalized Explanations

The Validate stage should compare each LLM output against its source `explanation_record`.

## Required Input

- `explanation_record`: JSON object following `revision_outputs/explanation_record_schema.json`.
- `llm_output`: JSON object with `local_explanation`, `evidence_used`, and `limitations`.

## Structural Checks

1. Output is valid JSON.
2. Output contains exactly the required top-level fields:
   - `local_explanation`
   - `evidence_used`
   - `limitations`
3. `local_explanation` is a non-empty string.
4. `evidence_used` is an array.
5. `limitations` is an array.

## Faithfulness Checks

1. Selected action check:
   - The selected action from `selected_action` must appear in `local_explanation`.
2. Alternative action check:
   - If `alternative_action` is not null, it must appear in `local_explanation`.
3. Positive contributor check:
   - Every contributor mentioned as beneficial/supporting must exist in `positive_contributors`.
   - No positive contributor may be described as a drawback.
4. Negative contributor check:
   - Every contributor mentioned as a drawback/trade-off must exist in `negative_contributors`.
   - No negative contributor may be described as beneficial unless the text explicitly says it is a trade-off.
5. Unsupported contributor check:
   - The explanation must not mention any component/feature absent from the record.
6. Numeric sign check:
   - Positive values must not be verbalized as negative.
   - Negative values must not be verbalized as positive.
7. Numeric value check:
   - If numeric values are included, they must match the record within a small formatting tolerance.
8. Total margin check:
   - If `total_margin` is mentioned, the value and sign must match the record.

## Stochasticity Checks

1. RWS / BTMM+RWS:
   - If `selection_mechanism` is `RWS` or `BTMM+RWS`, the text must not claim that the selected action was necessarily the deterministic best action unless the evidence supports this.
2. Low probability:
   - If `selected_probability` is present and low, e.g. less than `0.05`, the text should describe the selection as stochastic/low-probability rather than a strong deterministic preference.
3. Epsilon-greedy:
   - If `selection_mechanism` is `epsilon-greedy`, the text may mention exploration only if the source record contains an exploration flag or stochasticity note.

## Source Traceability Checks

1. `source_file` must be present in the record.
2. `source_row_id` must be present in the record.
3. The generated explanation should be logged with:
   - record id;
   - prompt version;
   - model metadata, if available;
   - validation pass/fail;
   - validation error messages.

## Fail Conditions

Reject the LLM output if any of the following occur:

- selected action is missing or changed;
- alternative action is missing when required;
- unsupported feature is introduced;
- contributor polarity is inverted;
- numeric sign is inverted;
- stochastic action is described as deterministic without support;
- output is not valid JSON;
- output omits `limitations` when required metadata is unavailable.
