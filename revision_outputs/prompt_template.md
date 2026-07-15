# Prompt Template for Offline LLM-Assisted Verbalization

This file records the exact implementation prompt and a shortened publication-facing prompt. The prompt applies only to LLM-assisted verbalization. Deterministic template rendering does not use this prompt, although rendered outputs can be checked against the same output schema and evidence-consistency rules.

## Exact Implementation Prompt

### System Prompt

You are an explanation verbalizer for reinforcement-learning experiments. You are an optional downstream language renderer only. You do not make decisions, recompute scores, change policy behavior, or infer hidden causes. Convert the provided structured numerical explanation record into concise human-readable text. Use only the supplied fields. Preserve action names, contributor names, numeric signs, and key values. If evidence is missing, say it is unavailable instead of inventing it.

### User Prompt Template

```text
Convert the following XRL explanation record into natural language.

Input record JSON:
{{EXPLANATION_RECORD_JSON}}

Requirements:
1. Use only evidence contained in the supplied record.
2. Write in English only.
3. Return local_explanation as a single string, not an array.
4. Mention the selected action exactly as provided, preserving spelling and capitalization.
5. Mention the alternative action exactly as provided if it is not null, preserving spelling and capitalization.
6. Never translate action names, contributor names, source field names, or numeric signs.
7. Preserve numerical values and their signs when you mention them.
8. Describe positive contributors only as supporting evidence.
9. Describe negative contributors only as trade-offs or limitations.
10. Do not introduce contributors that are not in the record.
11. Distinguish deterministic selection, exploration, and stochastic sampling using only the record fields.
12. If selection_mechanism is RWS or BTMM+RWS, describe the selected action as sampled/probabilistic rather than deterministically best unless the record explicitly supports a highest-scoring claim.
13. If selected_probability is low, state that the action was possible because of stochastic sampling.
14. If negative_contributors is empty, do not invent a trade-off sentence.
15. Return exactly these three JSON fields: local_explanation, evidence_used, limitations.
16. Use empty arrays when evidence_used or limitations has no entries.
17. Do not return Markdown fences or text outside the JSON object.
18. Keep the explanation concise.

Return valid JSON only in this shape:
{{
  "local_explanation": "one short paragraph or 3-4 bullets",
  "evidence_used": ["list of source fields used"],
  "limitations": ["missing or ambiguous fields, if any"]
}}

```

## Shortened Appendix Prompt

```text
Constrained prompt for LLM-assisted verbalization only

Given the structured explanation record below, generate a concise local policy explanation using only the supplied evidence.

Requirements:
- Preserve the selected action exactly.
- Preserve the alternative action exactly when it is available.
- Preserve numerical values and signs when mentioned.
- Describe positive contributors as supporting evidence.
- Describe negative contributors as trade-offs or limitations.
- Do not introduce contributors absent from the record.
- Distinguish deterministic selection, exploration, and stochastic sampling.
- Do not claim that a sampled action is the highest-scoring action unless the record explicitly supports that claim.
- Return exactly this JSON object, with empty arrays when no entries are available:

{
  "local_explanation": "string",
  "evidence_used": ["string"],
  "limitations": ["string"]
}

Structured explanation record:
{{EXPLANATION_RECORD_JSON}}

The prompt template applies only to LLM-assisted verbalization. The output schema and evidence-consistency requirements are shared across rendering modes.
```

## Output Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Explanation rendered output",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "local_explanation",
    "evidence_used",
    "limitations"
  ],
  "properties": {
    "local_explanation": {
      "type": "string",
      "minLength": 1
    },
    "evidence_used": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "limitations": {
      "type": "array",
      "items": {
        "type": "string"
      }
    }
  }
}
```

## Final Local Configuration Used for the Reported Full Run

- Provider: Ollama.
- Interface: Ollama local REST API.
- Endpoint: `http://localhost:11434`.
- Model: `qwen2.5:7b`.
- JSON generation mode: Ollama `format = "json"`.
- Maximum output length: `max_output_tokens = 600`, passed to Ollama as `num_predict = 600`.
- Temperature: not explicitly supplied; runtime default used.
- Top-p: not explicitly supplied; runtime default used.
- Seed: not supplied.
- Timeout: `60` seconds per HTTP request.
- Retry behavior: no automatic retry implemented.
- Failure handling: generation errors raise an exception; parse failures are retained as raw text and reported by validation; validation failures are logged for review and are not automatically accepted or automatically replaced.

The LLM step is only an Explain-stage verbalization layer. It is not used for RL training, policy learning, reward decomposition, action scoring, action selection, BTMM/RWS computation, or quantitative metric evaluation.
