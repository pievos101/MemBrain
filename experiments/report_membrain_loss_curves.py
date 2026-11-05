"""Render Membrain loss-curve benchmark results without rerunning model training."""
import csv
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/workspace/.cache/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/.cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import membrain_loss_curves as m

OUT=Path(__file__).resolve().parents[1]/'results/membrain_loss_curves'
TITLES={'null':'Untrained control','early':'Early training','mixed_difficulty':'Mixed text difficulty',
        'overfit_control':'Overfit positive control','natural_text':'Shakespeare (characters)'}


def main():
    rows=list(csv.DictReader((OUT/'metrics.csv').open()))
    paired=list(csv.DictReader((OUT/'paired_comparisons.csv').open()))
    def find(s,g,method):return next(r for r in rows if r['scenario']==s and r['group']==g and r['method']==method)
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(17,9),sharex=True)
    for ax,scenario in zip(axes.flat,m.SCENARIOS):
        rs=[find(scenario,'all',method) for method in m.METHODS]
        x=np.array([float(r['auc']) for r in rs]);lo=np.array([float(r['ci_low']) for r in rs]);hi=np.array([float(r['ci_high']) for r in rs])
        ax.errorbar(x,np.arange(7),xerr=[np.maximum(x-lo,0),np.maximum(hi-x,0)],fmt='o',capsize=3,color='#0072B2')
        ax.set_yticks(range(7),m.METHODS,fontsize=8);ax.invert_yaxis();ax.axvline(.5,ls=':',color='gray')
        ax.set_xlim(.35,1.02);ax.set_title(TITLES[scenario]);ax.set_xlabel('All-member ROC AUC (95% interval)')
    axes.flat[-1].axis('off');save(fig,'auc_comparison')
    comparisons=['Membrain auxiliary curve minus LOSS full text',
                 'Membrain paired curves minus LOSS full text',
                 'Membrain paired curves minus Original curve MLP']
    fig,axes=plt.subplots(1,3,figsize=(16,4),sharey=True)
    for ax,comparison in zip(axes,comparisons):
        rs=[next(r for r in paired if r['scenario']==s and r['group']=='all' and r['comparison']==comparison) for s in m.SCENARIOS]
        x=np.array([float(r['delta_auc']) for r in rs]);lo=np.array([float(r['ci_low']) for r in rs]);hi=np.array([float(r['ci_high']) for r in rs])
        ax.errorbar(x,np.arange(5),xerr=[np.maximum(x-lo,0),np.maximum(hi-x,0)],fmt='o',capsize=3)
        ax.axvline(0,color='gray',ls=':');ax.set_yticks(range(5),[TITLES[s] for s in m.SCENARIOS])
        ax.set_title(comparison.replace(' minus ','\nminus '),fontsize=10);ax.set_xlabel('Paired AUC difference (95% interval)')
    axes[0].invert_yaxis();save(fig,'paired_gains')
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,scenario in zip(axes.flat,m.SCENARIOS):
        records=[]
        for seed in m.TEST_SEEDS:
            with np.load(OUT/f'model_{scenario}_{seed}.npz') as f:records.append({k:f[k] for k in f.files})
        for name,key,sel,color,ls in [('Ordinary original','original',slice(0,200),'#0072B2','-'),
            ('Strong original','original',slice(200,240),'#009E73','-'),
            ('Nonmember original','original',slice(240,640),'#D55E00','-'),
            ('Strong generated','generated',slice(200,240),'#009E73','--'),
            ('Nonmember generated','generated',slice(240,640),'#D55E00','--')]:
            values=np.stack([(r[key][sel].mean(0) if key=='original' else r[key][sel].mean((0,1))) for r in records])
            avg,sd=values.mean(0),values.std(0,ddof=1)
            ax.plot(m.PREFIXES,avg,ls,marker='o',color=color,label=name)
            ax.fill_between(m.PREFIXES,avg-sd,avg+sd,color=color,alpha=.07)
        ax.set_title(TITLES[scenario]);ax.set_xticks(m.PREFIXES);ax.set_xlabel('Prefix length');ax.set_ylabel('Mean continuation NLL (nats/token)')
    handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].axis('off');axes.flat[-1].legend(handles,labels,loc='center')
    save(fig,'continuation_loss_curves')
    fig,axes=plt.subplots(2,3,figsize=(14,8));grid=np.linspace(0,1,101)
    for ax,scenario in zip(axes.flat,m.SCENARIOS):
        with np.load(OUT/f'predictions_{scenario}.npz') as f:
            for method in ['Membrain auxiliary curve','Membrain paired curves','LOSS same windows','LOSS full text']:
                curves=[]
                for p in f[method]:
                    positive,negative=p[:240],p[240:];ts=np.r_[np.inf,np.unique(p)[::-1],-np.inf]
                    fp=np.array([(negative>=t).mean() for t in ts]);tp=np.array([(positive>=t).mean() for t in ts])
                    u=np.unique(fp);upper=np.array([tp[fp==v].max() for v in u]);curves.append(np.interp(grid,u,upper))
                ax.plot(grid,np.mean(curves,0),label=method)
        ax.plot([0,1],[0,1],':',color='gray');ax.set_title(TITLES[scenario]);ax.set_xlabel('False positive rate');ax.set_ylabel('True positive rate')
    h,l=axes.flat[0].get_legend_handles_labels();axes.flat[-1].axis('off');axes.flat[-1].legend(h,l,loc='center');save(fig,'roc_comparison')
    text='''# Membrain using scalar continuation-loss curves

## Main result

Across these five settings, scalar-curve Membrain does not demonstrate a reliable
gain over conventional LOSS. The original/generated auxiliary-curve variant has
a small positive point estimate in mixed difficulty, but its paired interval
includes zero. The direct paired-curve variant shows no statistically resolved
gain over the original-curve-only MLP in any setting. Generated continuations
therefore have no demonstrated incremental benefit in these runs.

On Shakespeare, the original/generated variant reaches AUC 0.553, the direct
paired variant 0.565, the original-only curve MLP 0.569, same-window LOSS 0.571,
and full-text LOSS 0.584. The auxiliary-curve variant trails full-text LOSS by
0.031 (paired 95% interval [-0.055, -0.009]); the paired variant's difference is
-0.018 [-0.040, 0.003]. These results concern the small character-LM experiment,
not large pretrained language models.

The curve plots explain a plausible limitation: much of the class separation
appears as a vertical shift in original continuation loss, which scalar LOSS
already captures. Generated curves are often similar across membership groups.
This is a descriptive observation, not proof that every loss-curve representation
lacks additional information.

## Question and implementation

For each document, Membrain now uses an **original continuation-loss curve** and
a **generated continuation-loss curve** across prefix lengths. Each curve is a
two-dimensional plot (prefix length versus scalar loss), not a trajectory of
high-dimensional embeddings. Loss is mean next-token negative log-likelihood,
in nats/token, under the untempered language model. No hidden embeddings are
provided to any attack classifier.

Two implementations distinguish the original Membrain idea from direct paired
membership supervision:

- **Membrain auxiliary curve:** a 4→16→1 MLP learns original versus generated
  scalar curves from independent auxiliary texts, without membership labels.
  Its observed-minus-mean-generated logit gap is then mapped to membership using
  independent shadow labels. This preserves the original/generated-class objective.
- **Membrain paired curves:** a small 8→16→1 MLP sees four original losses and
  four mean generated losses together and learns membership on shadow models.
  This tests a direct two-curve variant, with membership supervision explicitly
  confined to shadows.

## Fair controls

- **LOSS full text:** negative mean NLL over all 47 next-token targets.
- **LOSS same windows:** negative mean over exactly the same 32 observed suffix
  targets seen by Membrain. This controls the observation locations and token budget.
- **Original curve MLP:** the same 16-hidden-unit architecture as paired Membrain,
  using only the four original losses. It tests whether generated continuations
  add value beyond a learned original-loss curve. Input parameter counts differ.
- **Generated-reference gap:** mean generated NLL minus mean observed NLL, followed
  by shadow fitting. This tests whether a simple local-difficulty reference suffices.
- **Min-K 20%:** negative mean of the ten highest token NLLs among 47 targets
  (20% rounded up). This is another scalar loss reference.

All three conventional LOSS/Min-K scores retain their predefined direction: lower
NLL means stronger membership evidence. Probability calibration has positive
scale and cannot reverse their rankings. Other score directions are learned on
shadows, never selected using target AUCs.

![AUC comparison](auc_comparison.png)
![Paired gains](paired_gains.png)

## Benchmark design

Five scenarios use the same GRU architecture (24-dimensional token embeddings,
48 hidden units, vocabulary 53), 48-unit sequences, prefixes 8/16/24/32, eight-unit
continuations and eight sampled generations per prefix. Sampling temperature is
1.0; generation losses are measured using the same raw LM probabilities as
observed losses. Generated losses are computed as each token is sampled, while
observed losses use the exactly corresponding next-token positions.

| Scenario | LM epochs | Strong exposures per epoch | Corpus |
|---|---:|---:|---|
| Untrained control | 0 | 0 | One Markov source; labels are nominal split labels |
| Early training | 3 | 4 | One shared Markov source |
| Mixed text difficulty | 3 | 4 | Easy/hard Markov sources, exactly balanced within every split |
| Overfit positive control | 30 | 15 | One shared Markov source |
| Natural text | 10 | 4 | Nonoverlapping Shakespeare character blocks |

Each scenario has four direction-training shadows, two separate probability
calibration models, and five untouched targets. MLP membership learners fit on
three shadows and use the fourth only for early stopping. Scalar learners fit
on all four shadows. Every target LM has independent auxiliary original/generated
training and validation texts. The auxiliary classifier never sees membership labels.

Per LM there are 400 ordinary training texts, 40 distinct repeated training texts,
400 evaluation nonmembers, 200 auxiliary-training and 80 auxiliary-validation
texts. Evaluation uses 200 ordinary members, all 40 repeated members and all 400
nonmembers. Exact duplicates are excluded across all splits. The mixed scenario
has exactly equal easy/hard proportions in every split; difficulty is not an
attack feature. Ordinary texts appear once per epoch. Markov texts are drawn
from the same source distribution for every membership class.

The natural-text scenario uses public Tiny Shakespeare from
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt.
The downloaded artifact SHA-256 is
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
The fixed alphabet is ASCII lowercase/uppercase letters plus a space; other
characters become spaces. Disjoint 48-character blocks are deduplicated, then
randomly assigned without replacement to splits. Prefix lengths here mean
characters. This is a natural-text sanity check with a small character LM,
not a modern pretrained-LLM benchmark. Blocks are disjoint within each LM;
different shadow and target LMs may sample some of the same public blocks.

The predefined five target seeds give 3,200 scored target documents per scenario.
Probabilities assume membership prior 240/640 = **0.375**, and a member mixture
of 5/6 ordinary and 1/6 repeated. A constant-prior predictor has Brier 0.234375.
Calibrators are frozen before target evaluation. Hyperparameters are fixed across
scenarios; these remain exploratory experiments rather than a preregistered study.

## Numerical results

AUCs average the five target models. Paired 95% intervals use 1,000 hierarchical
bootstrap replicates over target models, then documents within each selected
model. Identical resampled observations are used for each method in a comparison.
Intervals condition on the shadow-trained learners and calibrators. No multiple
comparison adjustment is applied. See CSVs for intervals on every individual AUC.

'''
    for scenario in m.SCENARIOS:
        text+=f'### {TITLES[scenario]}\n\n| Method | All AUC | Ordinary AUC | Repeated AUC | TPR at 5% FPR | Brier |\n|---|---:|---:|---:|---:|---:|\n'
        for method in m.METHODS:
            a,o,s=[find(scenario,g,method) for g in ['all','ordinary','strong']]
            text+=f'| {method} | {float(a["auc"]):.3f} | {float(o["auc"]):.3f} | {float(s["auc"]):.3f} | {float(a["tpr_at_5pct_fpr"]):.3f} | {float(a["brier"]):.3f} |\n'
        text+='\nPaired comparisons on all members:\n\n'
        for r in paired:
            if r['scenario']==scenario and r['group']=='all':
                text+=f'- {r["comparison"]}: {float(r["delta_auc"]):+.3f} [{float(r["ci_low"]):+.3f}, {float(r["ci_high"]):+.3f}].\n'
        text+='\n'
    text+='''
## Interpretation rules and limits

A convincing gain requires outperforming both full-text LOSS and information-
matched controls on held-out models. Outperforming one weak learned classifier
does not demonstrate improvement over LOSS. A paired-curve advantage over the
original-only MLP would support the generated-reference contribution; a gain over
the scalar mean alone may merely reflect learning weights across positions.
Neither comparison alone establishes that curve order is necessary.

The untrained control should remain at chance despite any separability of
original/generated curves. A saturated overfit result is a positive control,
not evidence of superiority. Difficulty variation is designed to test whether
generated continuations normalize intrinsic text predictability, while keeping
membership independent of easy/hard source assignment.

TPR at 5% FPR is a descriptive ROC statistic using an empirical threshold from
each target's 400 negatives, not a deployable threshold chosen without target
labels. No extreme low-FPR claim is supported. Mean ROC plots interpolate curves
per model before averaging. Continuation-curve shading shows one SD of target-
model means, not a confidence interval. A small calibration error alone does not
establish useful discrimination.

The benchmarks remain small and mostly synthetic. Shakespeare is character-level,
and exact deduplication does not remove semantically similar text. Repetition is
deliberately artificial. Only one horizon, temperature, sample budget and neural
architecture were tested. Generation requires more model calls than scalar LOSS;
no compute-efficiency advantage is claimed. There is no representation dimension
or token-probability information beyond scalar continuation NLL in these classifiers.
These experiments do not establish general performance on pretrained LLMs.

## Files and reproduction

`metrics.csv` contains all AUCs, intervals, Brier scores, ECE and TPR summaries;
`paired_comparisons.csv` contains paired gains. Per-model NPZ files preserve loss
curves, token losses, auxiliary gap and training histories. `learners_*.json`
preserve fitted standardization, MLP parameters and probability calibration.
`predictions_*.npz` contain all target membership probabilities.

Figures (PNG and PDF): `auc_comparison`, `paired_gains`,
`continuation_loss_curves`, `roc_comparison`.

From the repository root using the prepared Python environment:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
/workspace/.venvs/membrain/bin/python -m unittest discover -s experiments -p 'test_*.py'
/workspace/.venvs/membrain/bin/python experiments/membrain_loss_curves.py
/workspace/.venvs/membrain/bin/python experiments/report_membrain_loss_curves.py
```

Python 3.12, NumPy 2.2.6, PyTorch 2.6.0+cpu and Matplotlib 3.10.8 were used.
The natural-text input is retained under `source/` and checksum-verified on use.
Per-model caches are reused; use a fresh `--output` after model/data code changes.
`--scenarios` can run a subset while retaining other scenarios' summary rows.
The report generator reads the default output directory. All original benchmark
scripts and prior results remain intact.
'''
    (OUT/'REPORT.md').write_text(text)
    print('Saved Membrain loss-curve report and four PNG/PDF figures.')


if __name__=='__main__':main()
