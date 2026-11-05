"""Reviewer-directed report: incremental information, common support, ablations."""
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
import membrain_beyond_loss as m

OUT=m.ROOT/'results/membrain_beyond_loss'
NAMES={'cnn':'Ordered CNN','flat':'Flattened MLP','shuffle':'Shuffled CNN',
       'pooled':'Pooled MLP','loss_origin':'Loss-curve origin'}


def main():
    def read(name):
        with (OUT/f'{name}.csv').open() as f:return list(csv.DictReader(f))
    metrics=read('conditional_metrics');paired=read('paired_increments');matched=read('loss_matched')
    def metric(epoch,name):return next(r for r in metrics if int(r['epochs'])==epoch and r['method']==name)
    def contrast(epoch,name):return next(r for r in paired if int(r['epochs'])==epoch and r['comparison']==name)
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    methods=['LOSS + CNN','LOSS + flat','LOSS + shuffle','LOSS + pooled',
             'LOSS + loss curves','LOSS + recovery','LOSS + statistics','LOSS + statistics + CNN',m.ANCHOR_PROBE]
    fig,axes=plt.subplots(1,3,figsize=(18,5.5),sharey=True)
    for ax,epoch in zip(axes,m.BUDGETS):
        rows=[contrast(epoch,name+' minus LOSS') for name in methods]
        value=np.array([float(r['delta_auc']) for r in rows]);lo=np.array([float(r['auc_low']) for r in rows]);hi=np.array([float(r['auc_high']) for r in rows])
        ax.errorbar(value,np.arange(len(rows)),xerr=[np.maximum(value-lo,0),np.maximum(hi-value,0)],fmt='o',capsize=3)
        ax.axvline(0,color='gray',ls=':');ax.set_yticks(range(len(rows)),methods)
        ax.set_xlabel('Held-out AUC increment over flexible LOSS\n(two-way bootstrap 95% interval)')
        ax.set_title(f'{epoch} epochs'+(' (primary)' if epoch==5 else ' (sensitivity)'))
    axes[0].invert_yaxis();save(fig,'conditional_increments')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for name,color in [('Raw LOSS','#777777'),('LOSS','#000000'),('LOSS + CNN','#0072B2'),
                       ('LOSS + statistics','#D55E00'),('LOSS + statistics + CNN','#009E73')]:
        rows=[metric(e,name) for e in m.BUDGETS]
        axes[0].plot(range(3),[float(r['auc']) for r in rows],'o-',color=color,label=name)
        axes[1].plot(range(3),[float(r['nll']) for r in rows],'o-',color=color,label=name)
    for ax in axes:ax.set_xticks(range(3),m.BUDGETS);ax.set_xlabel('Sentence-exposure training epochs');ax.legend(fontsize=8)
    axes[0].set(ylabel='Mean held-out target ROC AUC',title='Does the representation add discrimination?')
    axes[1].set(ylabel='Held-out membership log loss (lower is better)',title='Does it add predictive information?')
    save(fig,'heldout_performance')
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for col,task in enumerate(['seen_vs_unseen','high_vs_low_exposure']):
        selected=[r for r in matched if r['task']==task and float(r['caliper'])==.02 and r['variant']=='cnn' and r['score']=='class1_logit']
        counts=[];wins=[];losswins=[]
        for epoch in m.BUDGETS:
            rows=[r for r in selected if int(r['epochs'])==epoch]
            ns=np.array([int(r['pairs']) for r in rows]);counts.append(ns.sum())
            valid=[r for r in rows if int(r['pairs'])>0]
            if valid:
                w=np.array([int(r['pairs']) for r in valid]);wins.append(np.average([float(r['score_winrate']) for r in valid],weights=w))
                losswins.append(np.average([float(r['loss_winrate']) for r in valid],weights=w))
            else:wins.append(np.nan);losswins.append(np.nan)
        axes[0,col].bar(range(3),counts,color='#888888');axes[0,col].set_ylabel('Unique matched pairs across four targets')
        axes[0,col].set_title(task.replace('_',' ')+': caliper 0.02 nats/character')
        axes[1,col].plot(range(3),wins,'o-',label='Membrain class-1 score')
        axes[1,col].plot(range(3),losswins,'s--',label='Raw LOSS, same matched pairs')
        axes[1,col].axhline(.5,color='gray',ls=':');axes[1,col].set_ylim(0,1.02)
        axes[1,col].set(ylabel='Within-pair concordance (descriptive)',xlabel='Training epochs')
        axes[1,col].set_xticks(range(3),m.BUDGETS);axes[1,col].legend(fontsize=8)
    save(fig,'loss_matched_support')
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,seed in zip(axes.flat,m.SEEDS['target']):
        with np.load(OUT/f'scores_{seed}_e5.npz') as f:r={k:f[k] for k in f.files}
        q=m.sigmoid(r['cnn'][:,0]);pairs=m.loss_match(r['loss'],r['repeats']>0,r['repeats']==0,.02)
        for a,b in pairs:ax.plot(r['loss'][[a,b]],q[[a,b]],color='#999999',alpha=.25,lw=.7)
        for level,color in zip([0,1,4,16],['#777777','#0072B2','#E69F00','#009E73']):
            take=r['repeats']==level
            ax.scatter(r['loss'][take],q[take],s=13,alpha=.65,color=color,label=f'{level*5} presentations')
        ax.set(title=f'Target {seed}: {len(pairs)} matched pairs',xlabel='Whole-sentence NLL (nats/character)',ylabel='Origin P(class 1: generated)')
    axes.flat[-1].legend(fontsize=8);save(fig,'loss_score_scatter')
    diagnostics=[]
    for path in OUT.glob('origin_metrics_*.json'):diagnostics+=json.loads(path.read_text())
    with (OUT/'origin_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(diagnostics[0]));w.writeheader();w.writerows(diagnostics)
    primary=contrast(5,'LOSS + CNN minus LOSS');stat=contrast(5,'CNN added to statistics')
    raw=contrast(5,'LOSS + CNN minus Raw LOSS')
    latent=contrast(5,m.EXTRA_PROBE+' minus LOSS')
    anchor=contrast(5,m.ANCHOR_PROBE+' minus LOSS')
    flat=contrast(5,'CNN versus flat');shuffle=contrast(5,'CNN versus shuffle')
    pdelta=float(primary['delta_auc']);plo=float(primary['auc_low']);phi=float(primary['auc_high'])
    verdict=('The primary conditional AUC increment is positive with an interval above zero.' if plo>0 else
             'The primary conditional AUC interval includes zero; an independent discrimination gain is not resolved.' if phi>=0 else
             'The primary conditional AUC increment is negative; adding the origin score reduces held-out discrimination.')
    text=f'''# Membrain reviewer-directed update: information beyond LOSS

## Primary result

{verdict}

At the prespecified middle budget (5 epochs), adding the ordered-CNN origin scores
to a flexible LOSS-only probe changes target AUC by **{pdelta:+.4f}**
(paired 95% interval **[{plo:+.4f}, {phi:+.4f}]**).
Against conventional raw negative-NLL LOSS, the increment is
{float(raw['delta_auc']):+.4f} [{float(raw['auc_low']):+.4f}, {float(raw['auc_high']):+.4f}].
Adding CNN scores on top of BOTH loss and recovery statistics changes AUC by
{float(stat['delta_auc']):+.4f} [{float(stat['auc_low']):+.4f}, {float(stat['auc_high']):+.4f}].

The ordered-versus-flattened contrast is {float(flat['delta_auc']):+.4f}
[{float(flat['auc_low']):+.4f}, {float(flat['auc_high']):+.4f}], and ordered versus
shuffled is {float(shuffle['delta_auc']):+.4f}
[{float(shuffle['auc_low']):+.4f}, {float(shuffle['auc_high']):+.4f}].
The exploratory 32-dimensional learned-representation probe changes AUC by
{float(latent['delta_auc']):+.4f} [{float(latent['auc_low']):+.4f}, {float(latent['auc_high']):+.4f}].
Those raw neuron coordinates are unaligned across independent origin nets and
are not a valid basis for ruling out representation information. A post-hoc,
coordinate-invariant auxiliary-anchor readout changes AUC by
{float(anchor['delta_auc']):+.4f} [{float(anchor['auc_low']):+.4f}, {float(anchor['auc_high']):+.4f}].

These are held-out-model representation tests. The origin classifier is still
trained without membership labels. The separate conditional probes **do** use
shadow membership labels and are evaluation instruments, not a relabeling of
Membrain's origin-class probabilities as membership probabilities.

![Conditional increments](conditional_increments.png)
![Held-out performance](heldout_performance.png)

## What changed in response to the reviewer

| Reviewer question | New experiment |
|---|---|
| Is this more than ordinary likelihood? | Flexible LOSS conditioning and held-out incremental AUC/log loss |
| Are histories distinguishable at similar loss? | Unique within-model loss matching at fixed calipers |
| Is a temporal CNN necessary? | Near-parameter-matched flattened MLP, identical-parameter shuffled CNN, pooled MLP |
| Do simple curves suffice? | True-target loss curves and exact/token/fixed-tail recovery statistics |
| Does auxiliary accuracy imply membership accuracy? | Separate auxiliary origin diagnostics from membership probe metrics |
| Was the old result saturated? | Fresh sentences and weaker 1/5/20-epoch checkpoints |

## Data and prospective protocol

All 576 sentences used by the previous final recovery/classifier study are
excluded. The 1,216 newly selected sentences are partitioned into disjoint pools:
128 LM background sentences, 320 auxiliary origin-class sentences, and 256 each
for shadow training, probe validation and target evaluation. Every LM uses the
same pool size and training recipe. Target sentences never train shadow LMs or
conditional probes. The corpus remains Shakespeare; this is not a new domain.

There are four shadow-training LMs, two validation LMs and four target LMs.
Each LM is checkpointed after 1, 5 and 20 sentence-training epochs following ten
background epochs. Repetition factors 0/1/4/16 yield total presentations:
0/1/4/16, 0/5/20/80, and 0/20/80/320, respectively. Each pool has 64 sentences
per factor per model; four-model rotations balance sentence identity across
exposures in the shadow and target groups. A checkpoint is a repeated measure
of an LM, not an additional independent LM.

The analysis plan was saved before training. Before conditional target results
existed, an amendment designated 5 epochs as primary and added raw LOSS as a
conventional comparator. The 1/20-epoch regimes and architecture/statistic
comparisons are sensitivity/exploratory analyses. This is locally recorded
prospective specification, not an external preregistration. Target results were
not used to choose regularization, calipers or classifier direction.

An additional frozen penultimate-layer (32-dimensional) CNN probe is exploratory:
it was added after the first result file was computed but before its outcomes
were inspected, and was not part of the prospective primary comparison. This
checks a richer learned representation rather than only the two score readouts.
After inspecting the primary results, a correctness issue was recognized: neuron
coordinates of independent origin classifiers are not aligned. An additional
exploratory anchor representation uses cosine similarities to 16 fixed true and
16 fixed sampled auxiliary-training trajectories. Relative geometry is invariant
to orthogonal neuron rotations/permutations and gives common anchor coordinates
across LMs. It uses no membership labels to construct features. This control is
post-hoc and does not alter the primary test; the earlier raw-coordinate probe
is retained as a failed diagnostic, not used as negative evidence about all
learned representation information.

## Origin-supervised representations and controls

Every frozen LM generates the ENTIRE remaining sentence at 12 increasing prefix
fractions (1%, 2%, 5%, then 10%…90%). Four sampled trajectories at temperature
1.0 and a greedy diagnostic trajectory are retained. All methods reuse those
model calls. The suffix state is 96-dimensional, giving a 12×96 trajectory.

The 320 auxiliary sentences split into 160 training, 40 early-stop, 40 class
calibration and 80 origin holdout. Class 0 is the true trajectory; class 1 the
generated trajectory. Class weighting is balanced at the document level.
No exposure/membership labels train normalization, origin nets or class calibration.

- **CNN:** two temporal convolutions and position-aware head (17,009 parameters).
- **Flattened MLP:** all 12×96 coordinates, one 15-unit hidden layer (17,311
  parameters; within 2% of CNN). Similar performance would weaken a claim that
  a temporal inductive bias is essential.
- **Shuffled CNN:** exactly the same parameters as CNN. Training independently
  shuffles rows of every trajectory; evaluation averages logits over eight fixed
  permutations. It tests availability of order, not merely a fixed relabeling of
  prefix positions. Finite permutation averaging is only approximately invariant.
- **Pooled MLP:** exact order-insensitive mean pooling over prefix rows.
- **Loss-curve origin control:** same auxiliary origin/conditioning labels, using
  scalar teacher-forced versus free-running true-target CE curves. It does not
  use hidden vectors. The generated-path loss is against the TRUE continuation,
  consistent with the clarified Membrain loss experiment, not generated self-likelihood.

## Conditional information test

The primary LOSS baseline is a piecewise-linear spline of whole-sentence NLL,
with nine knots set at shadow-training deciles. This allows a nonlinear
membership/likelihood relationship. Each augmented probe adds two origin scores:
the calibrated class-1 logit on true text and the true-minus-mean-generated
class-probability gap. A conventional **Raw LOSS** comparator preserves the
predefined lower-NLL-is-more-member ranking using positive calibration.

Loss-curve controls use per-prefix teacher-forced CE and mean free-running CE
against the true continuation (24 scalar features). Recovery controls use
sampled whole/fixed-tail exact match, sampled character accuracy, and greedy
whole-suffix exact match (60 scalar features). Statistics contain no hidden
vectors. Both are also combined before adding CNN scores as a stronger control.

Regularized logistic probes fit on four shadow LMs. Penalties 0.001, 0.01 and
0.1 are selected by membership log loss on two separate validation LMs; no target
fitting or calibration occurs. Membership prevalence is 75%. These supervised
probes test incremental information, while the representation learner remains
origin-supervised. They are not evidence of a fully label-free membership attack.

Intervals use 1,000 paired two-way bootstrap replicates over four target models
and shared target sentence IDs. The same draws are used for competing methods.
They condition on fitted shadow/probe parameters and do not include uncertainty
from rerunning the full selection/training pipeline. Four target models give
limited precision. Secondary comparisons have no multiplicity adjustment.

### Held-out metrics

| Epochs | Method | AUC [95% interval] | Membership log loss | Brier |
|---:|---|---:|---:|---:|
'''
    for epoch in m.BUDGETS:
        for name in ['Raw LOSS','LOSS','LOSS + CNN','LOSS + flat','LOSS + shuffle','LOSS + loss curves',
                     'LOSS + recovery','LOSS + statistics','LOSS + statistics + CNN',m.EXTRA_PROBE,m.ANCHOR_PROBE]:
            r=metric(epoch,name)
            text+=f'| {epoch} | {name} | {float(r["auc"]):.4f} [{float(r["ci_low"]):.4f}, {float(r["ci_high"]):.4f}] | {float(r["nll"]):.4f} | {float(r["brier"]):.4f} |\n'
    text+='\n### Primary-budget increments\n\n| Addition to flexible LOSS | AUC increment [95% interval] | Log-loss improvement [95% interval] |\n|---|---:|---:|\n'
    for name in methods:
        r=contrast(5,name+' minus LOSS')
        text+=f'| {name} | {float(r["delta_auc"]):+.4f} [{float(r["auc_low"]):+.4f}, {float(r["auc_high"]):+.4f}] | {float(r["nll_improvement"]):+.4f} [{float(r["nll_low"]):+.4f}, {float(r["nll_high"]):+.4f}] |\n'
    text+='''
## Similar-loss comparisons

Matching is done separately within each target LM, using only labels and absolute
whole-sentence NLL distance. Globally closest available pairs are selected
greedily without replacement; this is not guaranteed maximum-cardinality matching.
The primary caliper is 0.02 nats/character (about 2% perplexity difference);
0.01 and 0.05 are fixed sensitivity checks. Matching never uses a Membrain score.
Seen-vs-unseen and high-vs-low exposure comparisons are both retained.

Matching only evaluates common support. It can discard highly memorized examples
whose losses do not overlap the comparison group. Pair counts and retained
exposure composition are therefore reported, including empty matches. Similar
loss does not guarantee exact equality: raw LOSS's concordance on the same pairs
is shown rather than assuming it is at chance. Concordance is descriptive, not
a significance test of conditional independence.

![Loss-matched support](loss_matched_support.png)
![Primary-target scores and matching](loss_score_scatter.png)

| Epochs | Comparison | Caliper | Pairs | CNN class-score concordance | Raw LOSS concordance |
|---:|---|---:|---:|---:|---:|
'''
    for epoch in m.BUDGETS:
        for task in ['seen_vs_unseen','high_vs_low_exposure']:
            for caliper in m.CALIPERS:
                rs=[r for r in matched if int(r['epochs'])==epoch and r['task']==task and float(r['caliper'])==caliper and r['variant']=='cnn' and r['score']=='class1_logit']
                valid=[r for r in rs if int(r['pairs'])>0];n=sum(int(r['pairs']) for r in rs)
                if n:
                    w=[int(r['pairs']) for r in valid]
                    score=f'{np.average([float(r["score_winrate"]) for r in valid],weights=w):.3f}'
                    loss=f'{np.average([float(r["loss_winrate"]) for r in valid],weights=w):.3f}'
                else:score=loss='unavailable'
                text+=f'| {epoch} | {task} | {caliper} | {n} | {score} | {loss} |\n'
    text+='''
## Positioning and literature

The contribution being tested is **origin-supervised learning of continuation
representations and whether it adds conditional information beyond likelihood**.
This update does not claim that prefix-dependent recall, stochastic extraction,
or hidden-state analysis of memorization is new.

Official ACL Anthology metadata and abstracts verify:

- Hayes et al. (2025), *Measuring memorization in language models via probabilistic
  extraction*, NAACL, DOI 10.18653/v1/2025.naacl-long.469. Its abstract explicitly
  contrasts greedy extraction with probabilistic repeated-query extraction.
- Haviv et al. (2023), *Understanding Transformer Memorization Recall Through
  Idioms*, EACL, DOI 10.18653/v1/2023.eacl-main.19. Its abstract describes
  internal-representation analysis of memorization recall.

The reviewer-provided Xie et al. dynamic-prefix-shortening and Ilić et al.
memorization-signature references are retained as **unverified pointers**, not
used as independently established facts. Direct publisher/arXiv pages returned
HTTP 403 under the environment policy. Full papers were not read. Metadata,
abstracts and verification status are saved in `related_work_verification.json`.
A complete novelty assessment needs those papers; the manuscript must not claim
priority while their overlap remains unresolved.

## Limits and reproducibility

This is still a small character-GRU, single-corpus study. Fresh within-study
target sentences/models do not establish cross-domain or pretrained-LLM validity.
Loss conditioning is finite-dimensional; no experiment proves complete statistical
independence from likelihood. Probe regularization and distribution shift can
affect increments. Small or absent common support is a limitation, not a positive
result. Auxiliary origin accuracy is not membership accuracy.

The analysis plan and amendment, role-specific sentence pools, all LM weights,
origin nets, raw features, raw matching pairs, fitted probes and target predictions
are retained. CSVs contain all three regimes, all architectures and all calipers;
none are selected after target inspection. Four figure families have PNG/PDF files.

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
python experiments/membrain_beyond_loss.py
python experiments/membrain_beyond_loss.py --analysis-only
python experiments/report_membrain_beyond_loss.py
python -m unittest discover -s experiments -p 'test_*.py'
```

The earlier recovery/classifier sentence manifests and checksum-verified source
must exist so exclusions can be verified. See `experiments/README.md` for setup.
Use a fresh output after changing model/feature definitions. The report reads
the default `results/membrain_beyond_loss` directory. No earlier results or
original benchmark files are overwritten.
'''
    (OUT/'REPORT.md').write_text(text)
    print('Saved reviewer-directed report and four PNG/PDF figure families.')


if __name__=='__main__':main()
