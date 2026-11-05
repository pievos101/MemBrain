"""Regenerate combined figure and report from completed benchmark CSVs."""
import csv
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/workspace/.cache/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/workspace/.cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
METHODS = ['LM loss', 'Mean affinity', 'Late affinity (p32)', 'Affinity slope', 'CNN original gap', 'Pooled original gap']


def main():
    data = {}
    for epochs, folder in [(5, 'corrected_5epochs'), (60, 'corrected')]:
        with (RESULTS/folder/'auc_summary.csv').open() as f:
            data[epochs] = list(csv.DictReader(f))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    lines = []
    for row, epochs in enumerate([5,60]):
        lines += [f'### {epochs} LM training epochs', '', '| Method | Ordinary AUC [95% interval] | Strong AUC [95% interval] |', '|---|---:|---:|']
        table = {}
        for col, group in enumerate(['ordinary','strong']):
            ax = axes[row,col]
            for condition, offset, color in [('untrained',-.12,'#999999'),('trained',.12,'#0072B2')]:
                rs = [next(r for r in data[epochs] if r['condition']==condition and r['group']==group and r['method']==m) for m in METHODS]
                values = np.array([float(r['auc']) for r in rs])
                low = np.array([float(r['ci_low']) for r in rs]); high = np.array([float(r['ci_high']) for r in rs])
                ax.errorbar(values,np.arange(len(METHODS))+offset,xerr=[np.maximum(values-low,0),np.maximum(high-values,0)],fmt='o',color=color,capsize=3,label=condition)
                if condition=='trained':
                    table[group] = [f'{v:.3f} [{l:.3f}, {h:.3f}]' for v,l,h in zip(values,low,high)]
            ax.axvline(.5,ls=':',color='black',lw=1); ax.set_xlim(0,1.02)
            ax.set_yticks(range(len(METHODS)),METHODS)
            ax.set_title(f'{epochs} epochs: {group} members vs nonmembers')
            if row==1: ax.set_xlabel('ROC AUC (95% hierarchical bootstrap interval)')
        for i,m in enumerate(METHODS):
            lines.append(f'| {m} | {table["ordinary"][i]} | {table["strong"][i]} |')
        lines.append('')
    axes[0,0].invert_yaxis(); axes[0,1].legend(loc='best')
    fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(RESULTS/f'corrected_overview.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    text = '''# Corrected continuation-trajectory experiments

## Main finding

The corrected synthetic benchmark provides evidence of membership information in
averaged continuation representations after substantial overfitting, but does not
show an advantage over negative LM loss, or establish an advantage from trajectory
order. The manuscript's positive original-minus-generated classifier score is
inversely associated with membership here. These are exploratory synthetic results,
not confirmation of the manuscript's performance claims.

![Comparison across training budgets](corrected_overview.png)

## Design

- Five independent seeds (20261–20265); CPU, fixed public vocabulary of 53 tokens.
- A common stochastic Markov source generates all documents, with 48 tokens each.
  Its transition matrix is sampled independently per seed. Documents are deduplicated
  before random split assignment. All split pairs are checked for exact overlap.
- Per seed: 80 ordinary LM members, 20 distinct strong members, 100 nonmembers,
  100 auxiliary training documents and 40 auxiliary validation documents. Strong
  members appear 15 times per epoch; ordinary members appear once. Auxiliary
  documents never train the LM. All 200 evaluation documents are scored once per
  condition. Evaluation members are intentionally LM training documents.
- GRU LM: 24-dimensional token embeddings, 48 hidden units, Adam at 0.003,
  batches of 32, gradient clipping at 1. No external dataset or pretrained model.
- Prefixes 8, 16, 24, 32; observed and generated suffixes both contain exactly
  eight tokens, with identical causal hidden-state pooling. Sixteen independent
  sampled continuations per prefix, temperature 1.15. The state after consuming
  each suffix token is pooled; this fixes the previous truncation/alignment issue.
- Initial untrained weights provide a negative control. The same document splits
  and LM initialization are used for the 5- and 60-epoch budgets. The 60-epoch run
  was examined first; the 5-epoch follow-up was added because performance saturated.
  This is an exploratory follow-up, not a preregistered comparison.
- Original/generated attacks use only auxiliary origin labels, never membership
  labels. Their early stopping uses auxiliary original/generated validation loss.
  Membership labels are used only for evaluation; score signs are not flipped
  after observing evaluation AUCs.

## Scores

Let `o[k]` be the observed continuation's pooled hidden state and `g[m,k]` the
corresponding sampled states. The affinity at prefix k is:

`a[k] = -mean_d((o[k,d] - mean_m g[m,k,d])^2 / (var_m g[m,k,d] + floor[k]))`.

The variance floor is 0.1 times the mean variance over auxiliary training
documents and hidden dimensions at that prefix, with a numerical minimum of 1e-6.
Higher affinity means a more typical continuation in this diagonal hidden-state
metric. It is a heuristic, not a calibrated probability or density estimate.

- **LM loss:** negative mean next-token cross entropy over all 47 predicted tokens.
- **Mean affinity:** mean of four prefix affinities; order insensitive.
- **Late affinity:** affinity at prefix 32 alone.
- **Affinity slope:** least-squares affinity slope over equally spaced prefix
  indices (higher means increasing affinity with context).
- **CNN original gap:** `P(original | observed) - mean P(original | generated)`.
  A small two-layer CNN implements the manuscript's scoring principle, rather
  than exactly reproducing the v6 architecture. Inputs are standardized using
  auxiliary training data only. The classifier uses one sampled trajectory per
  auxiliary document; evaluation averages 16 generated trajectories.
- **Pooled original gap:** same objective and scoring with a shared per-prefix
  MLP and mean pooling, making the classifier invariant to prefix order. CNN and
  pooled models are not parameter matched, so their contrast is diagnostic only.

## Results

AUCs below are averaged across seeds. Intervals use 1,000 hierarchical bootstrap
replicates: resample seeds, then member and nonmember documents within each seed.
Paired comparisons reuse the same observations for both scores. Five seeds give
limited precision, and intervals are exploratory without multiplicity correction.
An empirical [1,1] interval reflects saturated observed rankings, not proof of
perfect performance on the population. Bootstrap intervals condition on the fitted
model within a sampled seed; they do not retrain models for each resample.

'''
    text += '\n'.join(lines)
    text += '''
## Interpretation and next experiment

At 60 epochs, mean affinity improves ordinary-member AUC over prefix 32 by
0.098 (paired 95% interval [0.056, 0.143]), but falls below loss by 0.274.
The slope is near chance, and the CNN-versus-pooled contrast includes zero.
Thus the benefit shown here is aggregation across observations; this experiment
does not establish that their temporal order contributes membership information.

At five epochs, mean affinity is near chance for ordinary and strong members.
Loss is already highly effective for the repeated texts (AUC 0.994). At 60 epochs
both affinity and loss saturate on those texts. The hidden-state metric may miss
changes in the output distribution that are already visible to likelihood.

The positive CNN gap gives AUC below 0.5 for strong members in both budgets.
This is consistent with memorized observed continuations looking more like
model-generated continuations, reducing their original-versus-generated gap.
It does not prove that mechanism. A reversed score would require validation on
independent shadow-model data; selecting its direction using these test results
would be exploratory. No reversed score is included in the primary tables.

Recommended next step: a larger, disjoint natural-text benchmark and independent
shadow-model calibration, retaining loss as the primary reference. If testing
trajectory order, use a parameter-matched classifier and explicit order ablations.
Do not tune against these evaluation splits. Low-false-positive-rate claims require
many more nonmembers than the 100 per seed used here.

Other limitations: the Markov corpus and repeated-exposure condition are artificial;
teacher-forced suffix hidden states require model internals (white-box access);
five seeds are not a broad robustness study; only a single generation temperature,
horizon and sample budget were tested. Loss sees all 47 next-token targets whereas
affinity uses four eight-token windows, so their information budgets differ. An
advantage over loss is not demonstrated under this setup. These results should
not replace the manuscript's numbers without rewriting its experimental design
and conclusions.

## Files and reproduction

- `corrected/`: 60 epochs, per-seed raw scores and LM training histories, configuration,
  AUC summaries, paired comparisons and three figures in PNG and PDF.
- `corrected_5epochs/`: the corresponding five-epoch experiment.
- `corrected_overview.png` / `.pdf`: combined comparison.
- Figures inside each folder: `auc_comparison`, `prefix_affinity`, `roc_curves`.
  ROC curves average per-seed interpolated ROC curves; they are descriptive.

From the repository root, using the prepared environment:

```bash
export MPLCONFIGDIR=/workspace/.cache/matplotlib
export XDG_CACHE_HOME=/workspace/.cache
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
/workspace/.venvs/membrain/bin/python -m unittest discover -s experiments -p test_corrected_benchmark.py
/workspace/.venvs/membrain/bin/python experiments/corrected_benchmark.py
/workspace/.venvs/membrain/bin/python experiments/corrected_benchmark.py --epochs 5 --output results/corrected_5epochs
/workspace/.venvs/membrain/bin/python experiments/report_corrected.py
```

Direct dependencies: Python 3.12, NumPy 2.2.6, PyTorch 2.6.0+cpu, Matplotlib 3.10.8.
For a fresh environment, install PyTorch from its verified CPU wheel index and
NumPy/Matplotlib from PyPI. `config.json` records actual experiment settings and
NumPy/PyTorch versions. Reruns overwrite results in the selected output directory;
choose a new `--output` directory to preserve previous results. The six original
benchmark files remain unchanged.
'''
    (RESULTS/'REPORT.md').write_text(text)
    print('Saved results/REPORT.md and corrected_overview.png/.pdf')


if __name__ == '__main__':
    main()
