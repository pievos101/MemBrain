"""Plots and report for Membrain's learned original/generated trajectory boundary."""
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
import membrain_trajectory_classifier as m

OUT=m.ROOT/'results/membrain_trajectory_classifier'
COLORS={'trajectory':'#0072B2','pooled':'#009E73','single90':'#E69F00'}


def main():
    aux=list(csv.DictReader((OUT/'auxiliary_holdout.csv').open()))
    exposure=list(csv.DictReader((OUT/'exposure_scores.csv').open()))
    membership=list(csv.DictReader((OUT/'membership_auc.csv').open()))
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    original=np.empty((4,128));sampled=np.empty((4,128));greedy=np.empty((4,128));seen=np.zeros((4,128),int)
    for seed in range(51001,51005):
        with np.load(OUT/f'predictions_{seed}.npz') as f,np.load(m.RECOVERY/f'model_{seed}.npz') as r:
            po=f['trajectory_p_generated_original'];pg=f['trajectory_p_generated_reference'].mean(1)
            gg=po-f['trajectory_greedy_gap']
            for j,level in enumerate(m.recovery.LEVELS):
                ids=np.where(r['repeats']==level)[0];seen[j,ids]+=1
                original[j,ids]=po[ids];sampled[j,ids]=pg[ids];greedy[j,ids]=gg[ids]
    assert np.all(seen==1)
    rng=np.random.default_rng(2183);draws=rng.integers(128,size=(1000,128))
    def mean_interval(values):
        bs=values[:,draws].mean(2)
        return values.mean(1),*np.quantile(bs,[.025,.975],axis=1)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));x=np.arange(4);counts=[0,40,160,640]
    for values,label,color in [(original,'True continuation trajectory','#0072B2'),
        (sampled,'Sampled-generated trajectory','#D55E00'),(greedy,'Greedy-generated trajectory (diagnostic)','#009E73')]:
        mu,lo,hi=mean_interval(values)
        axes[0].plot(x,mu,'o-',color=color,label=label);axes[0].fill_between(x,lo,hi,color=color,alpha=.12)
    for values,label,color in [(original-sampled,'True minus sampled reference','#0072B2'),
                               (original-greedy,'True minus greedy reference','#009E73')]:
        mu,lo,hi=mean_interval(values)
        axes[1].plot(x,mu,'o-',color=color,label=label);axes[1].fill_between(x,lo,hi,color=color,alpha=.12)
    axes[0].set_ylabel('Classifier P(class 1: generated)');axes[0].set_ylim(0,1.02)
    axes[0].set_title('Membrain: true texts move toward the generated region')
    axes[1].set_ylabel('Paired class-1 probability gap');axes[1].axhline(0,color='gray',ls=':')
    axes[1].set_title('The true/generated score gap closes with exposure')
    for ax in axes:ax.set_xticks(x,counts);ax.set_xlabel('Training presentations of the same sentence');ax.legend(fontsize=8)
    save(fig,'class_scores_by_exposure')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for variant,color in COLORS.items():
        means=[];sds=[]
        for count in counts[1:]:
            values=[float(r['auc']) for r in membership if r['variant']==variant and r['score']=='class1_probability' and int(r['presentations'])==count]
            means.append(np.mean(values));sds.append(np.std(values,ddof=1))
        means=np.array(means);sds=np.array(sds)
        axes[0].plot(np.arange(3),means,'o-',label=variant,color=color)
        axes[0].fill_between(np.arange(3),np.maximum(0,means-sds),np.minimum(1,means+sds),alpha=.1,color=color)
    for score,label in [('class1_probability','Class-1 score on true text'),('paired_gap','Sampled-reference gap'),
                        ('greedy_gap','Greedy-reference gap (diagnostic)'),('LOSS_full_sentence','LOSS')]:
        values=np.array([[float(r['auc']) for r in membership if r['variant']=='trajectory' and r['score']==score and int(r['presentations'])==count] for count in counts[1:]])
        axes[1].plot(np.arange(3),values.mean(1),'o-',label=label)
    loss=np.array([np.mean([float(r['auc']) for r in membership if r['variant']=='trajectory' and r['score']=='LOSS_full_sentence' and int(r['presentations'])==count]) for count in counts[1:]])
    axes[0].plot(np.arange(3),loss,'k--',label='LOSS')
    for ax in axes:
        ax.set_xticks(np.arange(3),counts[1:]);ax.set_xlabel('Seen-sentence presentations (versus unseen)')
        ax.set_ylim(.4,1.02);ax.axhline(.5,color='gray',ls=':');ax.legend(fontsize=8)
    axes[0].set_ylabel('Membership ROC AUC, mean of four LMs')
    axes[0].set_title('Trajectory versus pooled and single-prefix boundaries')
    axes[1].set_title('Membrain scores and LOSS; no membership fitting')
    save(fig,'membership_association')
    fig,ax=plt.subplots(figsize=(6,5));grid=np.linspace(0,1,101)
    import torch
    for variant,color in COLORS.items():
        curves=[]
        for seed in range(51001,51005):
            checkpoint=torch.load(OUT/f'classifier_{seed}_{variant}.pt',weights_only=True)
            with np.load(OUT/f'features_{seed}.npz') as r:
                po,pg=m.predict(checkpoint,r['original'][240:320],r['generated'][240:320,1:])
            positive=pg.ravel();negative=po;threshold=np.r_[np.inf,np.unique(np.r_[positive,negative])[::-1],-np.inf]
            fp=np.array([(negative>=t).mean() for t in threshold]);tp=np.array([(positive>=t).mean() for t in threshold])
            u=np.unique(fp);upper=np.array([tp[fp==v].max() for v in u]);curves.append(np.interp(grid,u,upper))
        a=np.mean([float(r['origin_auc']) for r in aux if r['variant']==variant])
        ax.plot(grid,np.mean(curves,0),color=color,label=f'{variant}: AUC {a:.3f}')
    ax.plot([0,1],[0,1],':',color='gray');ax.set(xlabel='False positive rate: true called generated',
        ylabel='True positive rate: generated called generated',title='Independent auxiliary origin-class holdout')
    ax.legend();save(fig,'auxiliary_origin_roc')
    text='''# Membrain: learned true/generated embedding-trajectory classifier

## Main finding

The learned trajectory boundary distinguishes true and sampled-generated
continuations with 96.9% balanced accuracy and 0.989 AUC on independent auxiliary
sentences. Without membership supervision, its class-1 score on true texts rises
from about 0.052 for unseen sentences to 0.453 after 640 presentations, while the
true/generated paired gap moves toward zero. This is the proposed learned
trajectory-convergence signal in the controlled recovery setting.

## What was built

This implements the proposed classifier directly:

- **Class 0:** true prefix followed by its true continuation.
- **Class 1:** the same true prefix followed by a model-generated suffix.

At 12 increasing prefix fractions (1%, 2%, 5%, 10%…90%), the LM generates the
entire remaining sentence. Hidden states after the prefix are mean-pooled into a
96-dimensional suffix vector. The true and generated sequences use the same
causal processing, prefix and suffix length. Each sample therefore produces a
12×96 true trajectory and one 12×96 trajectory per generated path.

The trajectory CNN learns the original/generated difference pattern across these
prefixes. It receives the raw suffix embedding trajectories, not recovery labels,
membership labels or training exposure counts. No scalar cosine or exact-match
measure is supplied as a shortcut.

![Class scores and convergence](class_scores_by_exposure.png)
![Auxiliary class discrimination](auxiliary_origin_roc.png)
![Association with membership](membership_association.png)

## Data separation and training

The four frozen character GRUs are the saved whole-sentence recovery models.
Their 128 evaluation sentences and 128 background sentences are excluded from
all classifier fitting, early stopping and calibration. A further 320 distinct
Shakespeare sentences (80–120 characters) form the origin-class experiment:
160 classifier training, 40 early stopping, 40 class-probability calibration and
80 final auxiliary holdout. These 320 auxiliary sentences never trained any LM.
The same auxiliary split is used across the four independently trained LMs, each
of which has its own origin classifier because its embedding coordinates differ.

Each training sentence supplies its true trajectory and four sampled-generated
trajectories. Class-weighted binary cross-entropy gives total weight 1/2 to each
class, rather than counting generated paths as independent extra documents.
Normalization uses classifier-training data only. Early stopping uses separate
sentences; a positive temperature and intercept are then fitted to the separate
calibration set. Calibration cannot reverse class ranking.

Eight sampled paths at temperature 1.0 are used for evaluation and reference
scores. Greedy generation is retained as a diagnostic reference but is not used
to fit or calibrate the sampled-origin classifier. Greedy-reference probabilities
are therefore outside the calibrated decoding distribution.

The CNN has two temporal convolutions (96→32→16 channels), then a position-aware
flattened head with 32 hidden units and dropout 0.1. Adam uses learning rate
0.001 and weight decay 0.01, with document batches of 32. A pooled-trajectory MLP
and a single-prefix-90% MLP provide descriptive controls. Their parameter counts
are not matched, so performance differences do not prove that temporal order
itself is the necessary signal.

## Class probabilities and scoring a new text

For a supplied text x, let H_true be its true-continuation trajectory and H_gen[m]
the trajectories generated from its successive prefixes. The saved classifier
returns:

`P0(x) = P(class 0: true continuation | H_true)`

`P1(x) = 1 - P0(x)`.

It also returns the optional paired score:

`gap(x) = P1(H_true) - mean_m P1(H_gen[m])`.

These are **origin-class probabilities** calibrated under a balanced true/generated
auxiliary prior. They are not calibrated probabilities of having been in LM
training. Membership is evaluated afterward as an association, without fitting
any membership labels or selecting a score direction on the evaluation set.
The predefined direction is that a true text looking more generated-like, or a
paired gap moving upward toward zero, gives stronger membership evidence.

If generated and true continuations become identical, their embedding
trajectories are identical and any deterministic classifier must give them the
same output despite their nominally different origin labels. This is expected.
The score-gap collapse can expose recall even when class-1 probability itself
does not approach one. A class label cannot distinguish two identical inputs.

## Numerical results

The following are means across four LMs. Origin holdout is independent of all
classifier training, stopping and calibration. It contains 80 true trajectories
and eight sampled trajectories per sentence; metrics weight the two classes
equally. No generated path is treated as an independently sampled document for
uncertainty claims.

| Classifier | Balanced origin accuracy | Origin ROC AUC | Balanced Brier |
|---|---:|---:|---:|
'''
    for v in m.VARIANTS:
        rs=[r for r in aux if r['variant']==v]
        text+=f'| {v} | '+' | '.join(f'{np.mean([float(r[k]) for r in rs]):.3f}' for k in ['balanced_accuracy','origin_auc','balanced_brier'])+' |\n'
    text+='\n### Trajectory-CNN scores on the recovery evaluation sentences\n\n| Presentations | P(class 1) on true text | P(class 1) on sampled reference | Paired sampled gap | Paired greedy gap |\n|---:|---:|---:|---:|---:|\n'
    for count in counts:
        rs=[r for r in exposure if r['variant']=='trajectory' and int(r['presentations'])==count]
        text+=f'| {count} | '+' | '.join(f'{np.mean([float(r[k]) for r in rs]):.3f}' for k in ['mean_p_generated_original','mean_p_generated_reference','mean_paired_gap','mean_greedy_gap'])+' |\n'
    text+='\n### Seen versus unseen association, trajectory classifier\n\n| Seen presentations | Class-1 score AUC | Sampled-gap AUC | Greedy-gap AUC (diagnostic) | LOSS AUC |\n|---:|---:|---:|---:|---:|\n'
    for count in counts[1:]:
        rs=[r for r in membership if r['variant']=='trajectory' and int(r['presentations'])==count]
        text+=f'| {count} | '+' | '.join(f'{np.mean([float(r["auc"]) for r in rs if r["score"]==score]):.3f}' for score in ['class1_probability','paired_gap','greedy_gap','LOSS_full_sentence'])+' |\n'
    text+='''
## Interpretation

The origin classifier learns a clear boundary on independent auxiliary texts.
On the evaluation sentences, increasing exposure makes true trajectories more
generated-like and closes their paired score gap. This is the proposed Membrain
mechanism, expressed as a learned classifier score rather than a hand-designed
embedding distance. The full trajectory is more informative than a single late
prefix in this experiment, while simple pooling captures much of the signal too.

This supports the construction of the origin classifier and an exposure-related
score here. It does not demonstrate superiority over LOSS: LOSS nearly perfectly
separates these deliberately exposed training sentences from unseen sentences.
The highest greedy-gap result is diagnostic and must not be interpreted as a
calibrated sampled-origin probability or an independently optimized attack.

## New-sample inference

The experimental scoring CLI uses the appropriate frozen LM and classifier pair:

```bash
/workspace/.venvs/membrain/bin/python experiments/membrain_trajectory_classifier.py \\
  --text-file results/membrain_trajectory_classifier/example_new_sentence.txt \\
  --model-seed 51001 --variant trajectory
```

The example is an independent auxiliary-holdout sentence, excluded from LM and
classifier training, early stopping and calibration. Output is also retained in
`example_prediction.json`. Inference accepts one printable-ASCII sentence of
80–120 characters, matching this trained experiment. It normalizes whitespace,
computes its true and generated trajectories, and returns P0, P1 and the paired
gap. No membership probability is invented from P1.

## Limits and artifacts

This remains a small character-GRU experiment with strong exposures of 40, 160
and 640 presentations. Known suffix length simplifies generation. There are only
four LMs, one corpus, one generation temperature and one sample budget. Exact
deduplication does not remove semantically similar sentences. Original trajectories
pool nested regions of one true sentence, while generated rows use independently
sampled continuations; cross-prefix consistency itself can help the origin task.
Its contribution to membership needs separate controls on richer models.

Bands on the class-score plot bootstrap the 128 sentence IDs, paired across
exposure conditions, conditional on these four fitted models and classifiers.
Membership plots show model means; shaded bands on classifier variants are one
SD across the four models, not a confidence interval. Scores were evaluated on
the already-inspected recovery benchmark, so this is exploratory follow-up work,
not an untouched confirmatory membership benchmark.

Saved classifier checkpoints include normalization, weights, labels and class
calibration. Feature NPZs retain complete trajectories. Prediction NPZs retain all
evaluation scores. CSVs contain auxiliary holdout metrics, exposure summaries and
per-model membership AUCs. Every plot has PNG and PDF versions.

To reproduce from the repository root in the prepared Python environment:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
/workspace/.venvs/membrain/bin/python -m unittest discover -s experiments -p 'test_*.py'
/workspace/.venvs/membrain/bin/python experiments/membrain_trajectory_classifier.py
/workspace/.venvs/membrain/bin/python experiments/report_membrain_trajectory_classifier.py
```

Python 3.12, NumPy 2.2.6, PyTorch 2.6.0+cpu and Matplotlib 3.10.8 were used.
Existing feature caches are reused. Use a fresh `--output` after changing feature
extraction; the report reads the default output directory. Prior experiments are
preserved. No classifier training changes the frozen LM weights.
'''
    (OUT/'REPORT.md').write_text(text)
    print('Saved Membrain trajectory-classifier plots and report.')


if __name__=='__main__':main()
