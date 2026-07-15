# LLM-assisted verbalization reproducibility manifest

This manifest preserves the rerun information referenced by the manuscript for the optional LLM-assisted verbalization layer.

## Scope

The LLM is used only in the Explain stage as a downstream language renderer over structured explanation records. It is not used for RL training, policy learning, reward decomposition, action scoring, action selection, BTMM/RWS computation, or quantitative metric evaluation.

## Local Ollama environment

- Provider: Ollama
- Interface: Ollama local REST API
- Endpoint: `http://localhost:11434`
- Ollama version: `0.32.0`
- Model: `qwen2.5:7b`
- Model digest: `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`
- Model size: `4683087332`
- Model family: `qwen2`
- Parameter size: `7.6B`
- Quantization: `Q4_K_M`
- Context length: `32768`

## Generation parameters

- JSON generation mode: Ollama `format = "json"`
- `num_predict = 600`
- `temperature = 0`
- `top_p = 1`
- `seed = 42`
- Timeout: `180` seconds per request
- Prompt version: `xrl-verbalizer-v1.0`
- Prompt record mode: `compact`

## Prompt and schema artifacts

- Prompt artifact: `revision_outputs/prompt_template.md`
- Explanation record schema: `revision_outputs/explanation_record_schema.json`
- Validation checks: `revision_outputs/llm_validation_checks.md`
- Generation script: `tools/generate_llm_outputs.py`
- Validation script: `tools/validate_verbalized_explanations.py`

## Rerun commands

Build normalized explanation records:

```powershell
& 'C:\Users\Acer\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tools\build_explanation_records.py --output revision_outputs\explanation_records.jsonl
```

Generate LLM verbalizations:

```powershell
& 'C:\Users\Acer\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tools\generate_llm_outputs.py --records revision_outputs\explanation_records.jsonl --output revision_outputs\llm_outputs_ollama_qwen25_7b.jsonl --base-url http://localhost:11434 --model qwen2.5:7b --num-predict 600 --temperature 0 --top-p 1 --seed 42 --timeout 180 --resume
```

Validate LLM verbalizations:

```powershell
& 'C:\Users\Acer\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tools\validate_verbalized_explanations.py --records revision_outputs\explanation_records.jsonl --llm-outputs revision_outputs\llm_outputs_ollama_qwen25_7b.jsonl --report revision_outputs\llm_validation_ollama_qwen25_7b_report.json
```

## Notes

- `--resume` allows interrupted generation to continue from existing rows in `revision_outputs/llm_outputs_ollama_qwen25_7b.jsonl`.
- The generation output JSONL stores `llm_metadata`, `generation_info`, `llm_output`, and the source file/row identifiers for each record.
- Only outputs that pass the validator should be used as accepted LLM-assisted explanations in manuscript examples.
