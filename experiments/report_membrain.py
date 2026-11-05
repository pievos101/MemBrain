"""Generate Membrain report and comparison plot from completed shadow experiments."""
import csv
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/workspace/.cache/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/.cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]/'results/membrain'
METHODS=['Membrain trajectory CNN','Membrain mean gap','Membrain prefix gaps','Membrain original region','LM loss']


def main():
    rows=list(csv.DictReader((OUT/'metrics.csv').open()))
    def find(e,m,g): return next(r for r in rows if int(r['epochs'])==e and r['method']==m and r['group']==g)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),sharey=True)
    for ax,group in zip(axes,['all','ordinary','strong']):
        for epochs,offset,color in [(5,-.13,'#E69F00'),(60,.13,'#0072B2')]:
            rs=[find(epochs,m,group) for m in METHODS]
            y=np.array([float(r['auc']) for r in rs]); lo=np.array([float(r['ci_low']) for r in rs]); hi=np.array([float(r['ci_high']) for r in rs])
            ax.errorbar(y,np.arange(len(METHODS))+offset,xerr=[np.maximum(y-lo,0),np.maximum(hi-y,0)],fmt='o',capsize=3,color=color,label=f'{epochs} epochs')
        ax.axvline(.5,ls=':',color='gray'); ax.set_xlim(0,1.02)
        ax.set_yticks(range(len(METHODS)),METHODS); ax.set_xlabel('Held-out target ROC AUC (95% interval)')
        ax.set_title(f'{group.capitalize()} members vs nonmembers')
    axes[0].invert_yaxis(); axes[-1].legend(); fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(OUT/f'membrain_performance.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    tables=[]
    for epochs in [5,60]:
        tables += [f'### {epochs} LM training epochs','','| Score | All-member AUC [95% CI] | Ordinary AUC | Strong AUC | Brier | ECE |','|---|---:|---:|---:|---:|---:|']
        for method in METHODS:
            a,o,s=[find(epochs,method,g) for g in ['all','ordinary','strong']]
            tables.append(f'| {method} | {float(a["auc"]):.3f} [{float(a["ci_low"]):.3f}, {float(a["ci_high"]):.3f}] | {float(o["auc"]):.3f} | {float(s["auc"]):.3f} | {float(a["brier"]):.3f} | {float(a["ece"]):.3f} |')
        tables.append('')
    stats=[]
    for epochs in [5,60]:
        rs=[]
        for seed in range(33001,33006):
            with np.load(OUT/f'model_{epochs}_{seed}.npz') as f: rs.append({k:f[k] for k in f.files})
        point=np.mean([r['aux_accuracy'] for r in rs]); joint=np.mean([r['joint_aux_accuracy'] for r in rs])
        stats.append(f'- {epochs} epochs: per-prefix original/generated validation accuracy {point:.3f}; whole-trajectory accuracy {joint:.3f} (target-model averages).')
    # Paired target bootstrap for trajectory CNN versus loss, conditional on the
    # frozen shadow-fitted direction and calibrator.
    paired=[]
    for epochs in [5,60]:
        with np.load(OUT/f'target_predictions_{epochs}.npz') as f:
            a,b=f[METHODS[0]],f['LM loss']
        from corrected_benchmark import auc
        rng=np.random.default_rng(871)
        def diff(i,p,n): return auc(a[i,p],a[i,n])-auc(b[i,p],b[i,n])
        est=np.mean([diff(i,np.arange(100),np.arange(100,200)) for i in range(5)])
        boot=[np.mean([diff(i,rng.integers(0,100,100),rng.integers(100,200,100)) for i in rng.integers(5,size=5)]) for _ in range(1000)]
        lo,hi=np.quantile(boot,[.025,.975])
        paired.append(dict(epochs=epochs,comparison='Membrain trajectory CNN minus LM loss',delta_auc=est,ci_low=lo,ci_high=hi))
    with (OUT/'paired_comparisons.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    text='''# Membrain: geometry and calibrated membership

This experiment tests the proposed missing link: whether a boundary learned from
original versus model-generated continuations can predict training membership on
independent target models. It separates region classification, membership direction
learning, probability calibration and target evaluation.

![Membrain held-out performance](membrain_performance.png)

## Protocol

At **each training budget** (5 and 60 epochs), there are 12 independent language
models: four shadow models for membership direction learning, three separate
models for probability calibration, and five held-out target models. Thus 24 LMs
are trained, with no target membership labels used to fit or calibrate predictions.
The same seeds across budgets provide matched corpus/initialization conditions.
The whole-trajectory CNN was added after viewing the per-prefix diagnostic, to
test dependencies a per-prefix boundary could miss. Comparisons are exploratory.

Every LM has a fresh corpus from the same stochastic Markov source family,
deduplicated before random assignment to disjoint sets. There are 80 ordinary
training members, 20 distinct strongly repeated members (15 exposures per epoch),
100 evaluation nonmembers, 100 auxiliary classifier-training documents and 40
auxiliary validation documents. The auxiliary texts never train the LM. Exact
document overlaps are asserted absent. The original six scripts are unchanged.

The LM and continuation settings match the corrected experiment: 53-token fixed
vocabulary, 48-token documents, a GRU with 24 embedding and 48 hidden dimensions,
prefixes 8/16/24/32, and eight-token observed/generated continuations. Sixteen
generations are sampled per prefix at temperature 1.15. Both continuation types
use identical causal processing and hidden-state pooling.

## What Membrain learns

**Region layer (no membership labels).** Four small per-prefix MLPs distinguish
original from generated continuation embeddings using auxiliary documents only.
The diagnostic coordinate is `log P(generated) - log P(original)`. Early stopping
uses independent auxiliary original/generated validation documents. This directly
tests the proposed original/generated embedding regions at each prefix.

**Whole-trajectory variant.** A two-layer CNN processes all four prefix embeddings
jointly. It uses the corrected benchmark's original/generated training procedure,
with training-only input standardization and auxiliary validation early stopping.
Its score is `P(generated | original trajectory) - mean P(generated | generated
trajectory)`. The CNN architecture is a compact implementation of the Membrain
principle, not an exact reproduction of the original v6 model.

**Membership layer (shadow membership labels).** Ridge-regularized logistic models
learn the sign and strength of region-score membership associations using only
the four training shadows. We evaluate:

- **Membrain trajectory CNN:** one whole-matrix region probability gap.
- **Membrain mean gap:** mean per-prefix logit gap, observed minus generated.
- **Membrain prefix gaps:** all four per-prefix logit gaps as logistic features.
- **Membrain original region:** mean observed per-prefix region logit alone.
- **LM loss:** negative mean next-token loss, with the same downstream fitting.

Standardization is fitted to training shadows only. A positive temperature and
intercept are then fitted on the three calibration models. This calibration
cannot reverse the learned score ranking. Five target models are evaluated only
after the direction learner and calibrator are frozen. There is no post-hoc test
score reversal or target-based method selection. This Membrain variant therefore
uses membership supervision at the shadow stage, although the region layer does not.

## Probability interpretation

The calibrated probability concerns a **50% membership prior**, with members
comprising 80% ordinary and 20% strongly repeated texts. This is the evaluation
mixture, not a universal prior for arbitrary user texts. Different membership
prevalence or training exposure can invalidate its calibration. Under a pure
prior shift from 0.5 to pi, odds can be adjusted as
`p_new = pi*p / (pi*p + (1-pi)*(1-p))`; this assumes the class-conditional score
distributions otherwise remain unchanged. Domain shift requires new validation.

Always predicting 0.5 gives Brier score 0.25 in this balanced setting. A small
calibration error by itself is not useful membership inference: a constant prior
can be calibrated while having AUC 0.5. ECE uses ten equal-width bins and is
descriptive; the reliability plot omits bins containing fewer than ten documents.

## Held-out results

AUC is averaged over the five target models. Its 95% interval uses 1,000
hierarchical bootstrap replicates over target models and then documents within
each model. It conditions on the fitted shadow learner and calibrator; it does
not incorporate uncertainty from retraining those two stages. Brier and ECE pool
the 1,000 target observations (500 members and 500 nonmembers).

'''
    text+='\n'.join(tables)
    text+='\n### What this says about Membrain\n\n'
    for epochs in [5,60]:
        a,o,s=[find(epochs,METHODS[0],g) for g in ['all','ordinary','strong']]
        text+=(f'- At {epochs} epochs, the full-trajectory Membrain classifier reaches '
               f'strong-member AUC {float(s["auc"]):.3f} '
               f'[{float(s["ci_low"]):.3f}, {float(s["ci_high"]):.3f}], '
               f'ordinary-member AUC {float(o["auc"]):.3f}, and all-member '
               f'Brier score {float(a["brier"]):.3f}.\n')
    text+='''
The full-trajectory boundary exposes a modest, transferable signal for strongly
repeated texts that the per-prefix plots largely miss. Ordinary-member intervals
include chance. This supports continuing to investigate joint continuation
geometry, but does not establish reliable membership classification or an
advantage from temporal order. The CNN trails LM loss at both training budgets.
Calibration correctly leaves substantial uncertainty rather than converting weak
region separation into confident membership claims. In these runs, the central
90% of full-trajectory membership probabilities span about 0.446–0.531 at five
epochs and 0.329–0.622 at 60 epochs, under the assumed 50% prior.

The next useful question is whether richer joint representations expose
membership information beyond likelihood on independent natural-text models.
Keep a frozen target split, compare against loss, and use a parameter-matched
order-insensitive control before attributing any benefit to trajectory order.

'''
    text+='\n### Auxiliary-task diagnostic\n\n'+'\n'.join(stats)
    text+='''

## How to read the geometry figures

- `region_trajectories_5.png` and `_60.png` show mean generated-region probability
  for member originals, nonmember originals, member-generated continuations and
  nonmember-generated continuations across prefix lengths. Shading is one SD of
  the five target-model means, not a confidence interval. Region-classifier
  probabilities are explicitly not membership probabilities.
- `region_distributions_5.png` and `_60.png` show decision-region probability
  distributions. The generated distribution gives one mean per document, rather
  than counting 16 dependent generations as 16 independent documents. Classifiers
  differ between models; these are aligned decision coordinates, not common raw
  embedding coordinates or proof of distinct embedding clusters.
- `membership_calibration_5.png` and `_60.png` show independently calibrated target
  membership probabilities and reliability. A curve near the diagonal with
  predictions concentrated at 0.5 means little useful discrimination.

## Interpretation limits

The per-prefix original/generated boundary is only useful if it captures a
membership-related distinction. Overlapping decision-region distributions mean
this particular boundary does not expose the proposed signal reliably. Strong
auxiliary original/generated accuracy, when present, is still not sufficient to
establish membership prediction. The whole-trajectory comparison tests a richer
boundary but is not a parameter-matched trajectory-order ablation.

This is a small synthetic GRU experiment using mean-pooled hidden states, one
generation temperature and one horizon. Negative results concern this
implementation and corpus; they do not disprove all possible Membrain
representations. The raw region logits are saved to support diagnosing changes.
There are only four direction-training and three calibration models, so shadow
generalization is limited. AUC confidence intervals are exploratory and have no
multiple-comparison adjustment. An empirical AUC of 1 does not prove perfect
population discrimination. Loss and Membrain observe different information
budgets, and all hidden-state variants require white-box access.

## Reproduction and files

From the repository root, using Python 3.12, NumPy 2.2.6, PyTorch 2.6.0+cpu and
Matplotlib 3.10.8 in the prepared environment:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
/workspace/.venvs/membrain/bin/python -m unittest discover -s experiments -p 'test_*.py'
/workspace/.venvs/membrain/bin/python experiments/membrain_geometry.py
/workspace/.venvs/membrain/bin/python experiments/report_membrain.py
```

The benchmark caches per-model raw region logits, auxiliary validation accuracy,
whole-trajectory gap and LM loss histories in `model_<epochs>_<seed>.npz`.
Reruns reuse these files; use `--output` with a new directory after changing model
code or experimental settings. Cached generation sample count and epoch count are
checked. The report generator reads the default `results/membrain` directory.
`calibrators_*.json` preserve fitted parameters; `target_predictions_*.npz`
preserve all target probabilities. `metrics.csv` and `paired_comparisons.csv`
contain numerical results. All figures also have PDF versions.
'''
    text+='\n### Paired whole-trajectory comparison with loss\n\n'
    for r in paired:
        text+=f'- {r["epochs"]} epochs: Membrain minus loss AUC = {r["delta_auc"]:.3f}, paired 95% interval [{r["ci_low"]:.3f}, {r["ci_high"]:.3f}].\n'
    (OUT/'REPORT.md').write_text(text)
    print('Saved Membrain report, performance plot and paired comparisons.')


if __name__=='__main__': main()
