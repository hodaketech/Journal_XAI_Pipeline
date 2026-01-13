
This repository contains the case studies used in the accompanying paper to evaluate the proposed taxonomy, the explanation pipeline (Decompose → Score → Select → Explain → Validate), and the standardized evaluation framework for policy explanations in Reinforcement Learning.
Repository Objectives
•	To provide source code for reproducing the three main case studies presented in the paper.
•	To automatically log evaluation results and store them as reports/Excel files or logs to support further analysis (e.g., faithfulness, sufficiency, compactness, etc.).

Requirements
•	Python 3.8+ (Python 3.9/3.10 is recommended)
•	Common libraries: numpy, pandas, scipy, openpyxl (for exporting Excel files), and other project-specific dependencies if required.

Directory Structure (Summary)
Maze_drQ_Epsilon_Greedy/
    Source code for Case Study 1 (DRQ + ε-greedy)

Maze_drQ_RWS/maze-XRL/
    Source code for Case Study 2 (DRQ + RWS)

Connect6/BitBoard/
    Source code for Case Study 3 (Connect6, BTMM + RWS)

Steps to Run the Code
1.	Clone the repository
2.	Install the required packages
3.	Run the case studies
Detailed instructions for each case study are provided below.
Each script supports two main modes:
•	evaluation
•	auto-validation
Case Study 1: Maze (DRQ + ε-greedy)
Directory: Maze_drQ_Epsilon_Greedy
Run Evaluation
•	If grid_size = 10:
cd Maze_drQ_Epsilon_Greedy
python drQ-main-with-evaluation.py maze10.txt
•	If grid_size = 20:
python drQ-main-with-evaluation.py maze20.txt
Run Validation (Automatic)
•	If grid_size = 10:
python drQ-main-auto-validation.py maze10.txt
•	If grid_size = 20:
python drQ-main-auto-validation.py maze20.txt
Output Notes
All results (metrics, data tables, and logs related to evaluation/validation) are logged and saved in the excel-results/ directory (as configured in the code).
Please check the Excel/CSV files in excel-results/ after execution.

Case Study 2: Maze (DRQ + RWS)
Directory: Maze_drQ_RWS/maze-XRL
Run Evaluation
•	If grid_size = 10:
python drQ-main-DRQ+RWS-with-evaluation.py maze10.txt
•	If grid_size = 20:
python drQ-main-DRQ+RWS-with-evaluation.py maze20.txt
Run Validation (Automatic)
•	If grid_size = 10:
python drQ-main-DRQ+RWS-auto-validation.py maze10.txt
•	If grid_size = 20:
python drQ-main-DRQ+RWS-auto-validation.py maze20.txt
Output Notes
All results are logged and stored in the excel-results/ directory, similar to Case Study 1.
Case Study 3: Connect6 (BTMM + RWS)
Directory: Connect6/BitBoard
Run Evaluation
python btmm_evaluation_main.py
Run Validation (Automatic)
python auto-validation.py
Output Notes
All evaluation results, traces, and validation logs for Connect6 are stored in the logs/ directory.
Brief Explanation of the Two Modes
•	Evaluation:
Collects evaluation data (e.g., Action Agreement, Sufficiency, Compactness, human-like metrics if applicable), generates tables/figures, and exports result files to excel-results/.
•	Auto-validation:
Performs automated validation checks (e.g., error codes MZ01.., MZ02.. for Maze; C6-01.. for Connect6 as defined in the paper) and records validation reports in excel-results/ or logs/.

