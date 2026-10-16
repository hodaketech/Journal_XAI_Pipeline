#!/usr/bin/env python3
"""User-supplied AES counting rules; outputs are written to the working directory."""
import argparse, json, re, sys
import pandas as pd

RULES = {
    "F1_selected_move_gamma": re.compile(r"chooses move\s+\d+.*?gamma\s*=", re.S),
    "F2_rws_mechanism":       re.compile(r"RWS"),
    "F3_best_move_compare":   re.compile(r"highest is move\s+\d+"),
    "F4_cumulative_prob":     re.compile(r"cumulative probability"),
    "F5_optimality_value":    re.compile(r"optimal move"),
}
LEVEL = re.compile(r"with a\s+(VERY LOW|VERY HIGH|MODERATELY HIGH|MODERATE|HIGH|LOW)\s+probability")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("--sheet", default=0)
    ap.add_argument("--col", default="human_explanation")
    a = ap.parse_args()
    df = pd.read_excel(a.path, sheet_name=a.sheet) if a.path.endswith(("xlsx","xls")) \
         else pd.read_csv(a.path, sep=None, engine="python")
    df = df.reset_index(drop=True); df["row"] = df.index + 2
    df = df.sort_values(["game_id","move_id"], kind="stable").reset_index(drop=True)
    n = len(df); s = n // 3; sizes = [s, (n - s)//2, n - s - (n - s)//2]
    df["phase"] = ["Opening"]*sizes[0] + ["Midgame"]*sizes[1] + ["Endgame"]*sizes[2]
    txt = df[a.col].fillna("").astype(str).str.strip()
    for k, rx in RULES.items(): df[k] = [int(bool(rx.search(t))) for t in txt]
    df["level"] = [(LEVEL.search(t).group(1) if LEVEL.search(t) else "") for t in txt]
    df["has_explanation"] = (txt != "")
    df["n_factors"] = df[list(RULES)].sum(axis=1).where(df.has_explanation)
    bad = df[df.has_explanation & ((df.F1_selected_move_gamma==0)|(df.F2_rws_mechanism==0))]
    if len(bad): print(f"WARNING: {len(bad)} explanations miss F1/F2 - check rows {bad.row.tolist()[:10]}", file=sys.stderr)
    ok = df[df.has_explanation]
    per_phase = {p: {"n_rows": int((df.phase==p).sum()),
                     "n_with_explanation": int(((df.phase==p)&df.has_explanation).sum()),
                     "aes": float(ok[ok.phase==p].n_factors.mean())} for p in ["Opening","Midgame","Endgame"]}
    out = {"n_rows": n, "n_empty_explanations": int((~df.has_explanation).sum()),
           "aes_record_mean": float(ok.n_factors.mean()),
           "aes_by_phase": per_phase,
           "aes_macro_over_phases": sum(v["aes"] for v in per_phase.values())/3,
           "level_counts": ok.level.value_counts().to_dict(),
           "factor_counts": {k:int(ok[k].sum()) for k in RULES}}
    df.to_csv("aes_cs3_records.csv", index=False); json.dump(out, open("aes_cs3.json","w"), indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
if __name__ == "__main__":
    main()
