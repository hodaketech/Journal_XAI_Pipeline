# Draft Offline LLM Verbalization Prompt

This is a proposed prompt template for future reproducible verbalization. It is not found in the repository and should not be presented as the original prompt unless the supervisor confirms it.

## System message

You convert structured numerical explanation evidence from reinforcement-learning experiments into concise human-readable explanations. Do not invent evidence. Use only the fields provided. Keep the selected action, compared alternative, top positive contributors, top negative contributors, and takeaway aligned with the input values.

## User message template

Generate a local explanation using this schema:

1. Selected action.
2. Compared alternative.
3. Top positive contributors and their numeric margins/probabilities.
4. Top negative contributors, if any.
5. One concise takeaway.

Input record:

```json
{
  "domain": "<Maze or Connect6>",
  "state_descriptor": "<coordinates, board phase, or compact state description>",
  "selected_action": "<action>",
  "alternative_action": "<action or null>",
  "positive_contributors": [
    {"name": "<component/feature>", "value": "<numeric value>", "meaning": "<short meaning>"}
  ],
  "negative_contributors": [
    {"name": "<component/feature>", "value": "<numeric value>", "meaning": "<short meaning>"}
  ],
  "decomposed_q_values": {
    "turn": "<value or null>",
    "goal": "<value or null>",
    "blocked": "<value or null>",
    "safe": "<value or null>"
  },
  "btmm_or_rws_scores": {
    "selected_probability": "<value or null>",
    "candidate_probabilities": "<dictionary or null>"
  },
  "total_margin": "<numeric margin or null>"
}
```

Constraints:

- Do not mention any component not present in the input.
- Do not change numeric signs.
- If a negative contributor is empty, omit the limitation sentence.
- Use one short paragraph or 3-4 bullets.
- Do not claim causality beyond the provided margins/probabilities.

## Output schema

```json
{
  "local_explanation": "<human-readable explanation>",
  "evidence_used": ["<field names used>"],
  "limitations": ["<missing or ambiguous fields>"]
}
```
