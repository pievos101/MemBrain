"""Plots/report for Membrain's confirmed true-target free-running CE definition."""
import csv
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/workspace/.cache/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/.cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import membrain_rollout as m

OUT=Path(__file__).resolve().parents[1]/'results/membrain_rollout'
TITLES={'null':'Untrained control','early':'Early training','mixed_difficulty':'Mixed difficulty',
        'overfit_control':'Overfit control','natural_text':'Shakespeare (characters)'}


def main():
    rows=list(csv.DictReader((OUT/'metrics.csv').open()))
    paired=list(csv.DictReader((OUT/'paired_comparisons.csv').open()))
    def find(s,g,method):return next(r for r in rows if r['scenario']==s and r['group']==g and r['method']==method)
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(18,10),sharex=True)
    for ax,scenario in zip(axes.flat,m.prev.SCENARIOS):
        rs=[find(scenario,'all',method) for method in m.METHODS]
        x=np.array([float(r['auc']) for r in rs]);lo=np.array([float(r['ci_low']) for r in rs]);hi=np.array([float(r['ci_high']) for r in rs])
        ax.errorbar(x,np.arange(len(rs)),xerr=[np.maximum(x-lo,0),np.maximum(hi-x,0)],fmt='o',capsize=3)
        ax.set_yticks(range(len(rs)),m.METHODS,fontsize=8);ax.invert_yaxis();ax.axvline(.5,ls=':',color='gray')
        ax.set_xlim(.35,1.02);ax.set_title(TITLES[scenario]);ax.set_xlabel('All-member ROC AUC (95% interval)')
    axes.flat[-1].axis('off');save(fig,'auc_comparison')
    comparisons=[('Membrain paired curves','LOSS full text'),('Membrain paired curves','LOSS same windows'),
                 ('Membrain paired curves','Original curve MLP')]
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
    for ax,(left,right) in zip(axes,comparisons):
        rs=[next(r for r in paired if r['scenario']==s and r['group']=='all' and r['comparison']==f'{left} minus {right}') for s in m.prev.SCENARIOS]
        x=np.array([float(r['delta_auc']) for r in rs]);lo=np.array([float(r['ci_low']) for r in rs]);hi=np.array([float(r['ci_high']) for r in rs])
        ax.errorbar(x,np.arange(5),xerr=[np.maximum(x-lo,0),np.maximum(hi-x,0)],fmt='o',capsize=3)
        ax.axvline(0,ls=':',color='gray');ax.set_yticks(range(5),[TITLES[s] for s in m.prev.SCENARIOS])
        ax.set_title(f'{left}\nminus {right}',fontsize=10);ax.set_xlabel('Paired AUC difference (95% interval)')
    axes[0].invert_yaxis();save(fig,'paired_gains')
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,scenario in zip(axes.flat,m.prev.SCENARIOS):
        records=[]
        for seed in m.prev.TEST_SEEDS:
            with np.load(OUT/f'model_{scenario}_{seed}.npz') as f:records.append({k:f[k] for k in f.files})
        for name,sel,color in [('Ordinary',slice(0,200),'#0072B2'),('Repeated',slice(200,240),'#009E73'),('Nonmember',slice(240,640),'#D55E00')]:
            for key,ls in [('teacher','-'),('rollout','--')]:
                values=np.stack([(r[key][sel].mean(0) if key=='teacher' else r[key][sel].mean((0,1))) for r in records])
                avg,sd=values.mean(0),values.std(0,ddof=1)
                ax.plot(m.prev.PREFIXES,avg,ls,marker='o',color=color,label=f'{name}: {key}')
                ax.fill_between(m.prev.PREFIXES,avg-sd,avg+sd,color=color,alpha=.06)
        ax.set_title(TITLES[scenario]);ax.set_xticks(m.prev.PREFIXES);ax.set_xlabel('Prefix length')
        ax.set_ylabel('CE against TRUE continuation (nats/token)')
    handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].axis('off');axes.flat[-1].legend(handles,labels,loc='center')
    save(fig,'true_target_curves')
    fig,axes=plt.subplots(2,3,figsize=(14,8));grid=np.linspace(0,1,101)
    for ax,scenario in zip(axes.flat,m.prev.SCENARIOS):
        with np.load(OUT/f'predictions_{scenario}.npz') as f:
            for method in ['Membrain auxiliary curve','Membrain paired curves','Membrain rollout mean','LOSS same windows','LOSS full text']:
                curves=[]
                for p in f[method]:
                    pos,neg=p[:240],p[240:];ts=np.r_[np.inf,np.unique(p)[::-1],-np.inf]
                    fp=np.array([(neg>=t).mean() for t in ts]);tp=np.array([(pos>=t).mean() for t in ts])
                    u=np.unique(fp);upper=np.array([tp[fp==v].max() for v in u]);curves.append(np.interp(grid,u,upper))
                ax.plot(grid,np.mean(curves,0),label=method)
        ax.plot([0,1],[0,1],':',color='gray');ax.set_title(TITLES[scenario]);ax.set_xlabel('False positive rate');ax.set_ylabel('True positive rate')
    handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].axis('off');axes.flat[-1].legend(handles,labels,loc='center');save(fig,'roc_comparison')
    text='''# Membrain: free-running cross-entropy against the true continuation

## Corrected definition

This experiment implements the user's confirmed definition. At each prefix p,
the LM receives the TRUE prefix and then follows its OWN sampled continuation.
At every continuation position, its next-token probability distribution is scored
against the TRUE token at that position. The true suffix is never fed back into
the free-running prediction history.

For sampled path y, the two loss curves are:

`L_rollout(p) = mean_j -log P(x[p+j] | x[:p], y[:j])`

`L_teacher(p) = mean_j -log P(x[p+j] | x[:p], x[p:p+j])`.

Indices in these expressions are zero-based. Both means cover the next eight
true tokens. Prefix lengths are 8, 16, 24 and 32. At each new prefix, generation
restarts from the true prefix. Eight paths are sampled independently per prefix,
at temperature 1.0. The sampled path includes a token at each position, but the
CE target is always the corresponding true token. Cross-entropy is evaluated
from full probability distributions, not from hard token equality.

**This is not the previous generated-token self-likelihood experiment.** Those
earlier files are preserved under `../membrain_loss_curves/`; none of their
generated scores are reused here. Teacher-forced losses and corpus/model seeds
are matched, so scoring changes can be compared without changing the LMs.
At the first suffix token both histories are identical; this equality is
asserted in every model run. Unit tests also force generation onto a known path
and verify exact CE against true tokens. Changing a true suffix token changes
its scored loss but not subsequent predictions on that fixed generated path.

![True-target curves](true_target_curves.png)
![AUC comparison](auc_comparison.png)
![Paired gains](paired_gains.png)

## Methods

- **Membrain auxiliary curve:** learns to distinguish teacher-forced and
  free-running scalar loss curves on independent auxiliary true continuations.
  No membership labels train this auxiliary classifier. Its observed-versus-
  mean-rollout logit gap is converted to membership using shadow labels.
- **Membrain paired curves:** an 8→16→1 MLP receives four teacher losses and four
  mean free-running true-target losses. Membership supervision uses shadows only.
- **Membrain rollout curve:** a 4→16→1 MLP sees only the free-running curve.
- **Membrain rollout mean:** negative mean free-running true-target CE, with a
  fixed lower-loss-is-more-member ranking.
- **Membrain excess mean:** rollout CE minus teacher CE, averaged across prefixes;
  its membership direction is learned on shadows.
- **Original curve MLP:** a 4→16→1 membership classifier using teacher losses only.
- **LOSS same windows:** negative mean teacher loss on the exact same 32 targets.
- **LOSS full text:** negative mean teacher loss over all 47 next-token targets.
- **Min-K 20%:** negative mean of the ten highest teacher token losses.

The LOSS and Min-K baselines retain their conventional predefined ranking.
All probability calibrations use positive scales, so calibration cannot reverse
score order. Generated paths and classifier inputs contain no hidden embeddings.
The paired/original/rollout MLPs share hidden width and fitting procedure, but
their input parameter counts differ. Prefix order is not separately ablated.

## Evaluation protocol

Five fixed scenarios retain the previous loss-curve benchmark's data and LM
settings: untrained control, early training (3 epochs, four repetitions), mixed
easy/hard difficulty (3 epochs, four repetitions), overfit control (30 epochs,
15 repetitions), and Shakespeare natural text (10 epochs, four repetitions).
Each scenario has four membership-training shadows, two calibration models and
five held-out targets, for 55 LM instances (44 trained and 11 untrained).

Each LM has 400 ordinary training texts, 40 repeated training texts, 400 evaluation
nonmembers and disjoint auxiliary sets of 200 training/80 validation texts.
Evaluation uses 200 ordinary and 40 repeated members plus 400 nonmembers.
Exact duplicates are excluded within each LM. Synthetic membership classes use
the same Markov source; mixed difficulty has identical source proportions in
every split. Null labels are nominal assignments: its LM receives no training.
Natural-text inputs are deduplicated, nonoverlapping 48-character Shakespeare
blocks with a fixed ASCII-letter-plus-space vocabulary. They are disjoint within
each LM; independent models may sample some of the same public blocks.

GRUs have 24 embedding and 48 hidden units, fixed vocabulary 53, Adam at 0.003,
batch size 32 and gradient clipping at 1.0. All model/data seeds, training budgets,
generation temperature and sample budget match the previous experiment. MLPs fit
on three shadows and early-stop on the fourth. Scalar learners fit on four
shadows. Two other models calibrate the probabilities; target labels fit nothing.

Probability calibration assumes prior 0.375 and a 5/6 ordinary, 1/6 repeated
member mixture. Constant-prior Brier score is 0.234375. This calibration does not
automatically transfer to different prevalence, corpora or exposure levels.

## Results

AUCs average five target models. Intervals use 1,000 paired hierarchical bootstrap
replicates over target models and documents, conditional on frozen shadow learners
and calibrators. Comparisons are exploratory with no multiplicity adjustment.
TPR at 5% FPR uses each target's empirical negative-score threshold as a descriptive
ROC statistic, not as an independently deployable operating threshold.

'''
    for scenario in m.prev.SCENARIOS:
        text+=f'### {TITLES[scenario]}\n\n| Method | All AUC | Ordinary AUC | Repeated AUC | TPR at 5% FPR | Brier |\n|---|---:|---:|---:|---:|---:|\n'
        for method in m.METHODS:
            a,o,s=[find(scenario,g,method) for g in ['all','ordinary','strong']]
            text+=f'| {method} | {float(a["auc"]):.3f} | {float(o["auc"]):.3f} | {float(s["auc"]):.3f} | {float(a["tpr_at_5pct_fpr"]):.3f} | {float(a["brier"]):.3f} |\n'
        text+='\nPaired all-member comparisons:\n\n'
        for r in paired:
            if r['scenario']==scenario and r['group']=='all':
                text+=f'- {r["comparison"]}: {float(r["delta_auc"]):+.3f} [{float(r["ci_low"]):+.3f}, {float(r["ci_high"]):+.3f}].\n'
        text+='\n'
    text+='''
## Findings and diagnosed limitation

The paired-curve variant does not demonstrate a positive all-member gain over
full-text LOSS in any scenario. On Shakespeare, free-running-curve-only Membrain
has AUC 0.510, paired-curve Membrain 0.566, same-window LOSS 0.571, and full-text
LOSS 0.584. The paired-versus-full LOSS difference is -0.017 with paired 95%
interval [-0.039, 0.004]. The rollout-only curve is significantly weaker than
the teacher-only curve in this character-LM setting; see the paired CSV.

The overfit auxiliary-curve result has a specific limitation: ordinary members
and heavily repeated members move in opposite directions relative to nonmembers.
For example, target seed 43001 has auxiliary gap AUC 0.228 for ordinary members
but 0.994 for repeated members. Its combined AUC is therefore below chance under
the single shadow-fitted monotonic membership mapping. The same pattern occurs
on training shadows; this is not simply a target-only score reversal. A perfect
or near-perfect teacher-versus-rollout auxiliary classification accuracy does
not solve this membership-mapping problem. No score is flipped after seeing
targets. A nonlinear auxiliary-score membership mapping could be a separate
future experiment; paired-curve MLPs already permit nonlinearity in both curves.

## Interpretation limits and reproduction

The rollout curve measures resistance to drifting away from the true continuation.
It can increase because generated history departs from truth even if the next
true token is intrinsically predictable under teacher forcing. Whether that drift
adds membership information is the empirical comparison here. A gain over
same-window LOSS is more informative than a gain over a weaker learned classifier.
Different prefix windows contain different true targets; these are matched local
continuations, not a fixed target span observed under increasing context.

The scenarios remain small, mostly synthetic and based on a GRU. Shakespeare
is character-level and does not represent a pretrained LLM attack. Strong exposure
is artificial; only one horizon, temperature and generation budget are tested.
Generated paths cost additional model calls, so no efficiency advantage is claimed.
Curve bands show SD across target-model means, not confidence intervals. AUC 1
or a narrow empirical interval does not prove perfect population discrimination.

From the repository root in the prepared Python environment:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
/workspace/.venvs/membrain/bin/python -m unittest discover -s experiments -p 'test_*.py'
/workspace/.venvs/membrain/bin/python experiments/membrain_rollout.py
/workspace/.venvs/membrain/bin/python experiments/report_membrain_rollout.py
```

Dependencies: Python 3.12, NumPy 2.2.6, PyTorch 2.6.0+cpu, Matplotlib 3.10.8.
The source text and checksum remain under the preceding benchmark's `source/`
directory. This run writes a separate `results/membrain_rollout/` directory.
Per-model `.pt` files preserve LM weights; `.npz` files preserve teacher losses,
free-running true-target losses, auxiliary scores and training histories.
`learners_*.json` and `predictions_*.npz` preserve fitted shadow learners and target
probabilities. `config.json` records the scoring definition and settings.
Use a fresh `--output` after model/scoring changes; old caches are checked for
the exact true-target scoring identifier and sample count before reuse.
All four figures have PNG and PDF versions. Prior results are untouched.
'''
    # Lead with the observed paired comparisons rather than a predetermined claim.
    summary='## Paired-curve result versus full-text LOSS\n\n'
    for scenario in m.prev.SCENARIOS:
        a=find(scenario,'all','Membrain paired curves');b=find(scenario,'all','LOSS full text')
        r=next(r for r in paired if r['scenario']==scenario and r['group']=='all' and r['comparison']=='Membrain paired curves minus LOSS full text')
        summary+=(f'- {TITLES[scenario]}: Membrain {float(a["auc"]):.3f}, LOSS {float(b["auc"]):.3f}; '
                  f'difference {float(r["delta_auc"]):+.3f} [{float(r["ci_low"]):+.3f}, {float(r["ci_high"]):+.3f}].\n')
    text=text.replace('## Corrected definition',summary+'\n## Corrected definition',1)
    (OUT/'REPORT.md').write_text(text)
    print('Saved true-target Membrain report and four PNG/PDF figures.')


if __name__=='__main__':main()
