## This repository contains the case studies used in the accompanying paper to evaluate the proposed taxonomy, the explanation pipeline (Decompose → Score → Select → Explain → Validate), and the standardized evaluation framework for policy explanations in Reinforcement Learning.
### Repository Objectives
- To provide source code for reproducing the three main case studies presented in the paper.
- To automatically log evaluation results and store them as reports/Excel files or logs to support further analysis (e.g., faithfulness, sufficiency, compactness, etc.).

### Requirements
•	Python 3.8+ (Python 3.9/3.10 is recommended)
•	Common libraries: numpy, pandas, scipy, openpyxl (for exporting Excel files), and other project-specific dependencies if required.

### Directory Structure
#### Maze_drQ_Epsilon_Greedy/
    Source code for Case Study 1 (reward-decomposed Q-values + ε-greedy)

#### Maze_drQ_RWS/maze-XRL/
    Source code for Case Study 2 (reward-decomposed Q-values + RWS)

#### Connect6/BitBoard/
    Source code for Case Study 3 (Connect6, BTMM + RWS)

#### Steps to Run the Code
1.	Clone the repository
2.	Install the required packages
3.	Run the case studies
Detailed instructions for each case study are provided below.
Each script supports two main modes:
- evaluation
- auto-validation
#### Case Study 1: Maze (reward-decomposed Q-values + ε-greedy)
Directory: Maze_drQ_Epsilon_Greedy

Run Evaluation
- If grid_size = 10:
cd Maze_drQ_Epsilon_Greedy
python drQ-main-with-evaluation.py maze10.txt
- If grid_size = 20:
python drQ-main-with-evaluation.py maze20.txt

Run Validation (Automatic)
- If grid_size = 10:
python drQ-main-auto-validation.py maze10.txt
- If grid_size = 20:
python drQ-main-auto-validation.py maze20.txt

Output Notes: All results (metrics, data tables, and logs related to evaluation/validation) are logged and saved in the excel-results/ directory (as configured in the code).
Please check the Excel/CSV files in excel-results/ after execution.

#### Case Study 2: Maze (reward-decomposed Q-values + RWS)
Directory: Maze_drQ_RWS/maze-XRL

Run Evaluation
- If grid_size = 10:
python drQ-main-DRQ+RWS-with-evaluation.py maze10.txt
- If grid_size = 20:
python drQ-main-DRQ+RWS-with-evaluation.py maze20.txt

Run Validation (Automatic)
- If grid_size = 10:
python drQ-main-DRQ+RWS-auto-validation.py maze10.txt
- If grid_size = 20:
python drQ-main-DRQ+RWS-auto-validation.py maze20.txt

Output Notes: All results are logged and stored in the excel-results/ directory, similar to Case Study 1.
#### Case Study 3: Connect6 (BTMM + RWS)
Directory: Connect6/BitBoard

Run Evaluation
python btmm_evaluation_main.py

Run Validation (Automatic)
python auto-validation.py

Output Notes: All evaluation results, traces, and validation logs for Connect6 are stored in the logs/ directory.
Brief Explanation of the Two Modes
- Evaluation:
Collects evaluation data (e.g., Action Agreement, Sufficiency, Compactness, human-like metrics if applicable), generates tables/figures, and exports result files to excel-results/.
- Auto-validation:
Performs automated validation checks (e.g., error codes MZ01.., MZ02.. for Maze; C6-01.. for Connect6 as defined in the paper) and records validation reports in excel-results/ or logs/.

#### Optional LLM-assisted verbalization

The LLM layer is a downstream renderer for structured explanation records. It is not used for RL training, reward decomposition, action scoring, action selection, BTMM/RWS computation, or quantitative metric evaluation.

Reproducibility artifacts are stored in:

- `revision_outputs/prompt_template.md`
- `revision_outputs/explanation_record_schema.json`
- `revision_outputs/llm_validation_checks.md`
- `revision_outputs/llm_reproducibility_manifest.md`

Local Ollama configuration used for the manuscript rerun:

- Ollama `0.32.0`
- model `qwen2.5:7b`
- model digest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`
- `num_predict=600`, `temperature=0`, `top_p=1`, `seed=42`

Build normalized records:

```powershell
python tools\build_explanation_records.py --output revision_outputs\explanation_records.jsonl
```

Generate LLM verbalizations:

```powershell
python tools\generate_llm_outputs.py --records revision_outputs\explanation_records.jsonl --output revision_outputs\llm_outputs_ollama_qwen25_7b.jsonl --base-url http://localhost:11434 --model qwen2.5:7b --num-predict 600 --temperature 0 --top-p 1 --seed 42 --timeout 180 --resume
```

Validate LLM verbalizations:

```powershell
python tools\validate_verbalized_explanations.py --records revision_outputs\explanation_records.jsonl --llm-outputs revision_outputs\llm_outputs_ollama_qwen25_7b.jsonl --report revision_outputs\llm_validation_ollama_qwen25_7b_report.json
```
