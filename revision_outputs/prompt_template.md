# Prompt Template for Offline LLM-Assisted Verbalization

This file records the exact implementation prompt and a shortened publication-facing prompt. The prompt applies only to LLM-assisted verbalization. Deterministic template rendering does not use this prompt, although rendered outputs can be checked against the same output schema and evidence-consistency rules.

## Exact Implementation Prompt

### System Prompt

You verbalize reinforcement-learning explanation records. Use only supplied evidence. Do not infer hidden causes, recompute scores, change actions, or claim sampled actions are deterministic best actions unless explicitly supported. Return valid JSON only.

### User Prompt Template

```text
Convert the following XRL explanation record into natural language.

Input record JSON:
{{EXPLANATION_RECORD_JSON}}

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

## Final Local Configuration for Reproducible Reruns

- Provider: Ollama.
- Interface: Ollama local REST API.
- Endpoint: `http://localhost:11434`.
- Model: `qwen2.5:7b`.
- Model digest: `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`.
- Ollama version: `0.32.0`.
- JSON generation mode: Ollama `format = "json"`.
- Maximum output length: `max_output_tokens = 600`, passed to Ollama as `num_predict = 600`.
- Temperature: `0`.
- Top-p: `1`.
- Seed: `42`.
- Timeout: `180` seconds per HTTP request.
- Prompt version: `xrl-verbalizer-v1.0`.
- Retry behavior: no automatic retry implemented.
- Failure handling: generation errors raise an exception; parse failures are retained as raw text and reported by validation; validation failures are logged for review and are not automatically accepted or automatically replaced.

The LLM step is only an Explain-stage verbalization layer. It is not used for RL training, policy learning, reward decomposition, action scoring, action selection, BTMM/RWS computation, or quantitative metric evaluation.
