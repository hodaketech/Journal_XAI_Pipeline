# btmm_explainability_summary.py

import os
import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --------- Config ----------
CSV_PATH = "evaluation-validation-dataset-processed.csv"
OUTPUT_PATH = "logs/btmm_evaluation_final.xlsx"
TOP_K = 1
SURROGATE_SEEDS = [0, 1, 2]
PERT_N = 10
PERT_NOISE_REL = 0.02
MAX_TREE_DEPTH = 8
RANDOM_STATE = 0
MIN_PER_CLASS_FOR_STRATIFY = 2

AES_MASS_THRESHOLD = 0.80


def safe_train_test_split(X, y, test_size=0.3, random_state=None, min_per_class=MIN_PER_CLASS_FOR_STRATIFY):
    if y is None or len(y) == 0:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) <= 1:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    if np.any(counts < min_per_class):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except Exception:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

def parse_pipe_list(s, dtype=float):
    if pd.isna(s):
        return []
    parts = str(s).split('|')
    if dtype == int:
        return [int(p) for p in parts if p != '']
    else:
        return [float(p) for p in parts if p != '']

def top_k_indices(gammas, k):
    if len(gammas) == 0:
        return []
    order = np.argsort(gammas)[::-1]
    return order[:k].tolist()

def compute_sufficiency(gammas, k):

    if len(gammas) == 0:
        return False
    sel = int(np.argmax(gammas))  # Original policy π(s)
    topk = top_k_indices(gammas, k)
    masked = np.zeros_like(gammas, dtype=float)
    masked[topk] = gammas[topk]  # Explanation-driven policy π_E(s)
    new_sel = int(np.argmax(masked)) if masked.sum() > 0 else -1
    return new_sel == sel

def compute_comprehensiveness(gammas, k):

    if len(gammas) == 0:
        return 0.0
    sel = int(np.argmax(gammas))  # Original action π(s)
    topk = top_k_indices(gammas, k)
    
    Q_original = gammas[sel] if 0 <= sel < len(gammas) else 0.0
     
    masked = np.array(gammas, dtype=float)
    masked[topk] = 0.0  # Remove cited factors E
    Q_removed = masked[sel] if 0 <= sel < len(masked) else 0.0
    
    return float(Q_original - Q_removed)

def compute_action_agreement(gammas, k):

    if len(gammas) == 0:
        return False
    original_policy = int(np.argmax(gammas))
    
    topk = top_k_indices(gammas, k)
    explanation_policy_vector = np.zeros_like(gammas, dtype=float)
    explanation_policy_vector[topk] = gammas[topk]
    
    if explanation_policy_vector.sum() == 0:
        return False
        
    explanation_policy = int(np.argmax(explanation_policy_vector))
    
    return original_policy == explanation_policy

def perturb_gammas(gammas, noise_relative=PERT_NOISE_REL, seed=None):
    rng = np.random.RandomState(seed)
    g = np.array(gammas, dtype=float)
    if g.sum() == 0:
        return g
    noise = rng.normal(loc=0.0, scale=noise_relative * (g.mean() + 1e-8), size=g.shape)
    g2 = g + noise
    g2 = np.clip(g2, a_min=0.0, a_max=None)
    if g2.sum() <= 0:
        return g / g.sum()
    return g2 / g2.sum()

def gammas_to_vector(gammas, length):
    v = np.zeros(length, dtype=float)
    for i, g in enumerate(gammas):
        if i < length:
            v[i] = g
    return v

def aes_by_mass(gammas, mass_threshold=AES_MASS_THRESHOLD):
    g = np.array(gammas, dtype=float)
    if g.size == 0 or g.sum() <= 0:
        return 0
    probs = g / g.sum()
    cum = np.cumsum(np.sort(probs)[::-1])  
    idx = int(np.searchsorted(cum, mass_threshold))
    return idx + 1

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    print("Loading CSV:", CSV_PATH)
    df = pd.read_csv(CSV_PATH, encoding='utf-8')

    if 'moves' not in df.columns or 'gamma_values' not in df.columns:
        raise RuntimeError("CSV need to have 'moves' and 'gamma_values' (pipe-separated).")

    print("Parsing moves and gamma_values...")
    df['moves_list'] = df['moves'].apply(lambda x: parse_pipe_list(x, dtype=int))
    df['gamma_list'] = df['gamma_values'].apply(lambda x: parse_pipe_list(x, dtype=float))

    def sel_idx(row):
        g = row['gamma_list']
        if len(g) == 0:
            return -1
        return int(np.argmax(g))
    df['selected_index'] = df.apply(sel_idx, axis=1)

    df = df.sort_values(['game_id', 'move_id']).reset_index(drop=True)
    print("Assigning groups (opening/midgame/endgame) by 1/3 split per game...")
    group_map = {}
    for gid, g in df.groupby('game_id'):
        idxs = g.index.tolist()
        n = len(idxs)
        if n == 0:
            continue
        b1 = int(math.floor(n / 3))
        b2 = int(math.floor(2 * n / 3))
        for pos_i, global_idx in enumerate(idxs):
            if pos_i < b1:
                group_map[global_idx] = 'opening'
            elif pos_i < b2:
                group_map[global_idx] = 'midgame'
            else:
                group_map[global_idx] = 'endgame'
    df['group'] = df.index.map(lambda i: group_map.get(i, 'midgame'))

    groups_list = ['opening', 'midgame', 'endgame']
    max_moves = int(df['moves_list'].apply(len).max()) if df.shape[0] > 0 else 0
    if max_moves <= 0:
        raise RuntimeError("No valid moves/gammas found in the file.")

    group_summaries = []
    for grp in groups_list:
        DF_grp = df[(df['group'] == grp) & (df['selected_index'] >= 0)].copy().reset_index(drop=True)
        n_states = len(DF_grp)
        print(f"Processing group '{grp}' with {n_states} states ...")
        if n_states == 0:
            group_summaries.append({
                'group': grp, 
                'n_states': 0,
                'AA': np.nan,                   
                'suff_frac': np.nan,             
                'comp_mean': np.nan,            
                'AES_mean': np.nan              
            })
            continue

        aa_bools = []        
        suff_bools = []    
        comp_drops = []      
        aes_vals = []      
        for _, row in DF_grp.iterrows():
            gammas = np.array(row['gamma_list'], dtype=float)
            if len(gammas) == 0:
                continue
            
            aa_bool = compute_action_agreement(gammas, TOP_K)
            aa_bools.append(int(aa_bool))
            
            s_bool = compute_sufficiency(gammas, TOP_K)
            suff_bools.append(int(s_bool))
            
            comp_drop = compute_comprehensiveness(gammas, TOP_K)
            comp_drops.append(float(comp_drop))

            aes_vals.append(aes_by_mass(gammas, mass_threshold=AES_MASS_THRESHOLD))

        AA = float(np.mean(aa_bools)) if len(aa_bools) > 0 else float('nan')
        suff_frac = float(np.mean(suff_bools)) if len(suff_bools) > 0 else float('nan')
        comp_mean = float(np.mean(comp_drops)) if len(comp_drops) > 0 else float('nan')
        aes = float(np.mean(aes_vals)) if len(aes_vals) > 0 else float('nan')

        group_summary = {
            'group': grp,
            'n_states': n_states,
            'AA': AA,                    
            'suff_frac': suff_frac,     
            'comp_mean': comp_mean,      
            'AES_mean': aes             
        }
        group_summaries.append(group_summary)

    print("Computing summary_all_states as mean of the 3 phases (opening/midgame/endgame)...")
    summary_by_group_df = pd.DataFrame(group_summaries)

    if summary_by_group_df.shape[0] == 0:
        all_summary = {
            'n_states': 0, 
            'AA': np.nan, 
            'suff_frac': np.nan, 
            'comp_mean': np.nan, 
            'AES_mean': np.nan
        }
    else:
      
        numeric_cols = [c for c in summary_by_group_df.columns if c != 'group']
      
        mean_vals = summary_by_group_df[numeric_cols].mean(axis=0, skipna=True).to_dict()

        all_summary = {
            'n_states': float(mean_vals.get('n_states', np.nan)),
            'AA': float(mean_vals.get('AA', np.nan)),
            'suff_frac': float(mean_vals.get('suff_frac', np.nan)),
            'comp_mean': float(mean_vals.get('comp_mean', np.nan)),
            'AES_mean': float(mean_vals.get('AES_mean', np.nan))
        }

    outdir = os.path.dirname(OUTPUT_PATH)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    print("Writing results to", OUTPUT_PATH)
    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        pd.DataFrame([all_summary]).to_excel(writer, sheet_name='summary_all_states', index=False)

        summary_by_group_df.to_excel(writer, sheet_name='summary_by_group', index=False)

    print("Done. Results saved to:", OUTPUT_PATH)
    print("\nSummary by group:\n", summary_by_group_df.to_string(index=False))
    print("\nSummary for all states (mean of phases):\n", pd.DataFrame([all_summary]).to_string(index=False))

if __name__ == "__main__":
    main()