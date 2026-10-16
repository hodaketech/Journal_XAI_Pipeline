from pathlib import Path
import ast
import hashlib
import json
import platform
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'revision_outputs' / 'cs3_aes_audit'
OUT.mkdir(exist_ok=True)
SOURCE = ROOT / 'Connect6/BitBoard/btmm_evaluation_main.py'
DATA = ROOT / 'Connect6/BitBoard/evaluation-validation-dataset-processed.csv'
names = {'top_k_indices', 'compute_sufficiency', 'compute_comprehensiveness',
         'compute_action_agreement', 'aes_by_mass'}
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
namespace = {'np': np, 'AES_MASS_THRESHOLD': .8}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                             and n.name in names], type_ignores=[]), str(SOURCE), 'exec'), namespace)
df = pd.read_csv(DATA)
df['source_csv_line'] = np.arange(len(df)) + 2
df = df.sort_values(['game_id', 'move_id']).reset_index(drop=True)
df['phase'] = ''
for _, group in df.groupby('game_id'):
    n = len(group)
    for pos, idx in enumerate(group.index):
        df.loc[idx, 'phase'] = 'opening' if pos < n//3 else 'midgame' if pos < 2*n//3 else 'endgame'
states, curves = [], []
for _, row in df.iterrows():
    g = np.array([float(v) for v in row.gamma_values.split('|')])
    moves = [int(v) for v in row.moves.split('|')]
    aligned = len(g) == len(moves) == row.num_moves
    assert len(g) and np.isfinite(g).all() and (g >= 0).all() and g.sum() > 0
    p = g / g.sum()
    ref = int(np.argmax(g))
    identity = dict(source_csv_line=int(row.source_csv_line), game_id=int(row.game_id),
                    move_id=int(row.move_id), phase=row.phase)
    aes = namespace['aes_by_mass'](g)
    comp = namespace['compute_comprehensiveness'](g, 1)
    E1 = namespace['top_k_indices'](g, 1)
    dp = p[ref] if ref in E1 else 0.
    states.append(dict(**identity, candidates=len(g), gamma_sum=g.sum(),
        reference_argmax_action=moves[ref] if aligned else None, executed_action_available=False,
        candidate_alignment_valid=aligned, recorded_num_moves=int(row.num_moves),
        AA=int(namespace['compute_action_agreement'](g, 1)),
        Suff=int(namespace['compute_sufficiency'](g, 1)), Comp_raw_gamma=comp,
        P_argmax=p[ref], delta_P_output_mask=dp, normalized_Comp=dp/p[ref],
        bound_holds=dp <= p[ref]+1e-12, AES80=aes, AES80_fraction=aes/len(g),
        entropy_nats=-np.sum(p[p>0]*np.log(p[p>0])),
        entropy_normalized=-np.sum(p[p>0]*np.log(p[p>0]))/np.log(len(g)) if len(g)>1 else 0.,
        max_tie_count=int(np.sum(g == g[ref]))))
    for k in [1, 3, 5, 10, 20, 46]:
        ke = min(k, len(g))
        stable = np.argsort(-g, kind='stable')[:ke]
        old = namespace['top_k_indices'](g, k)
        for arm, suff, c, d in [
            ('legacy_top', float(namespace['compute_sufficiency'](g,k)), namespace['compute_comprehensiveness'](g,k), p[ref]*int(ref in old)),
            ('consistent_ties_top', float(ref in stable), g[ref]*int(ref in stable), p[ref]*int(ref in stable)),
            ('uniform_random_exact', ke/len(g), g[ref]*ke/len(g), p[ref]*ke/len(g))]:
            curves.append(dict(**identity,k=k,effective_k=ke,arm=arm,Suff=suff,
                               Comp_raw_gamma=c,delta_P_output_mask=d,normalized_Comp=d/p[ref]))
s = pd.DataFrame(states)
c = pd.DataFrame(curves)
s.to_csv(OUT/'per_row.csv',index=False)
c.to_csv(OUT/'curves_per_row.csv',index=False)
cols=['AA','Suff','Comp_raw_gamma','P_argmax','delta_P_output_mask','normalized_Comp',
      'AES80','AES80_fraction','candidates','entropy_nats','entropy_normalized','max_tie_count']
phase=s.groupby('phase')[cols].mean().reindex(['opening','midgame','endgame'])
phase.insert(0,'n_rows',s.groupby('phase').size())
phase.to_csv(OUT/'phase_summary.csv')
summary=c.groupby(['phase','k','arm'])[['Suff','Comp_raw_gamma','delta_P_output_mask','normalized_Comp','effective_k']].mean()
summary.to_csv(OUT/'curves_summary.csv')
clean=s.candidate_alignment_valid & s.gamma_sum.between(.99,1.01)
s[clean].groupby('phase')[cols].agg(['mean','count']).to_csv(OUT/'quality_filtered_summary.csv')
checks={}
checks['AA_equals_Suff_all_rows']=bool((s.AA==s.Suff).all())
checks['all_legacy_failures_are_tied']=bool((s.loc[s.Suff==0,'max_tie_count']>1).all())
checks['consistent_ties_Suff_always_one']=bool((c.loc[c.arm=='consistent_ties_top','Suff']==1).all())
checks['legacy_suff_is_reference_inclusion']=bool(np.allclose(c.loc[c.arm=='legacy_top','Suff'],c.loc[c.arm=='legacy_top','normalized_Comp']))
for gg in [np.array([.2,.5,.3]),np.array([.4,.4,.2]),np.array([0.,.6,.4])]:
    a=int(np.argmax(gg))
    for k in range(1,len(gg)+1):
        vals=[]
        for subset in itertools.combinations(range(len(gg)),k):
            keep=np.zeros_like(gg);keep[list(subset)]=gg[list(subset)]
            drop=gg.copy();drop[list(subset)]=0
            vals.append((float(keep.sum()>0 and np.argmax(keep)==a),gg[a]-drop[a]))
        assert np.allclose(np.mean(vals,axis=0),[k/len(gg),gg[a]*k/len(gg)])
checks['random_expectations_exhaustive_small_cases']=True
assert all(checks.values())
workbook=pd.read_excel(ROOT/'Connect6/BitBoard/logs/btmm_evaluation_final.xlsx',sheet_name=None)
old=workbook['summary_by_group'].set_index('group')
comparison={key:float(np.max(np.abs(phase[col]-old[key]))) for key,col in
            [('AA','AA'),('suff_frac','Suff'),('comp_mean','Comp_raw_gamma'),('AES_mean','AES80')]}
manifest=dict(python=platform.python_version(),numpy=np.__version__,
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE,DATA]},
    n_rows=len(s),distinct_game_ids=int(df.game_id.nunique()),
    distinct_game_move_keys=int(len(df[['game_id','move_id']].drop_duplicates())),
    exact_duplicate_rows=int(df.drop(columns=['source_csv_line','phase']).duplicated().sum()),
    AA_Suff_mismatches=int((s.AA!=s.Suff).sum()),bound_violations=int((~s.bound_holds).sum()),
    candidate_alignment_failures=int((~s.candidate_alignment_valid).sum()),
    quality_filtered_n=int(clean.sum()),checks=checks,
    top1_failures=int((s.Suff==0).sum()),failures_without_max_tie=int(((s.Suff==0)&(s.max_tie_count==1)).sum()),
    gamma_sum_range=[float(s.gamma_sum.min()),float(s.gamma_sum.max())],
    phase_macro=phase[cols].mean().to_dict(),row_micro=s[cols].mean().to_dict(),
    workbook_max_abs_difference=comparison,workbook={k:v.to_dict('records') for k,v in workbook.items()},
    missing=['executed RWS action/draw','board snapshots','active feature matrix','Q and visits traces',
             'source for AES 1.457','manuscript'],
    random_control='Exact expectation of uniform size-min(k,m) subsets without replacement; no seed needed.')
(OUT/'audit_summary.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
for j,ph in enumerate(['opening','midgame','endgame']):
    for arm,label,style in [('legacy_top','Legacy top-k','o-'),('consistent_ties_top','Consistent ties','s--'),('uniform_random_exact','Random (exact mean)','^:')]:
        z=summary.loc[(ph,slice(None),arm),:].reset_index()
        axes[0,j].plot(z.k,z.Suff,style,label=label)
        axes[1,j].plot(z.k,z.delta_P_output_mask,style,label=label)
    axes[0,j].set_title(ph.capitalize())
    for ax in axes[:,j]:
        ax.set_xlabel('k (capped at candidate count)');ax.grid(alpha=.2)
    axes[0,j].set_ylim(-.02,1.06)
axes[0,0].set_ylabel('Sufficiency (argmax agreement)')
axes[1,0].set_ylabel('Comp: normalized-gamma drop')
axes[0,0].legend(fontsize=8)
fig.suptitle('Historical output masking: sensitivity to k and tie handling')
fig.savefig(OUT/'suff_comp_k.png',dpi=180)
plt.close(fig)
fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
for ph in ['opening','midgame','endgame']:
    z=s[s.phase==ph];ax.scatter(z.P_argmax,z.delta_P_output_mask,label=ph,alpha=.55,s=18)
ax.plot([0,1],[0,1],'k--',linewidth=1,label='Delta P = P')
ax.set(xlabel='P of reference argmax (normalized recorded gamma)',ylabel='Delta P under legacy top-1 output masking',title='Reference action is not the recorded RWS draw')
ax.legend();fig.savefig(OUT/'comp_vs_probability.png',dpi=180);plt.close(fig)
print(json.dumps(manifest,indent=2))
print(phase.to_string())
