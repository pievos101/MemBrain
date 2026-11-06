# Membrain: unsupervised membership scoring

This is now the primary **Membrain** method. The previous shadow classifier
remains archived as a supervised experiment; it is NOT used here. The attack
has no fitting stage: no membership labels, origin classifier, shadow models,
cross-text feature normalization, learned score combination, threshold fitting,
or membership-probability calibration. Target LMs still train normally; known
membership labels are necessary to construct/evaluate a benchmark, never to
compute or choose the attack score. The output is a ranking score, not a
membership probability or an automatically calibrated binary decision.

## Fixed score, declared before evaluation

Retain the ten increasing TRUE-prefix fractions [.05,.1,...,.9], generating
the entire remaining sample greedily without teacher forcing. Compare hidden
states AFTER aligned true/generated tokens. Keep the original pooled cosine
as a diagnostic, but do not use a fitted origin classifier.

For each true-token position, run SPW-MIA adaptive shortening and obtain
`g_t = log p_full(true_t) - log p_short(true_t)`. Within each suffix use
`w = 0.5 * uniform + 0.5 * softmax(clip(g,-20,20))`. These are fixed choices.

At each prefix r:

- `a(r)` = weighted mean aligned true/generated hidden cosine.
- `b(r)` = the same cosine against true hidden states circularly shifted by
  half the suffix length, a deterministic within-text alignment-null control.
- `C(r) = clip((a-b)/max(1-b,1e-6),0,1)` removes generic embedding similarity.
  Collapsed identical embeddings yield no alignment evidence.
- `L_true(r)` = weighted teacher-forced TRUE-continuation token CE.
- `L_roll(r)` = weighted CE against TRUE tokens while conditioning on the
  model's own generated tokens. This is not generated-token self-likelihood.
- `E(r) = max(L_roll-L_true,0)/(1+L_true)` is relative rollout degradation.
- `R(r) = (0.5+0.5*C(r))/(1+E(r))`. A bounded embedding factor avoids treating
  imperfect greedy recovery as conclusive nonmembership. No coefficient is fit.

`A(R)` is normalized trapezoidal integration against density `(1-r)` over the
tested [.05,.9] interval: earlier recovery earns more support. Let `H` be mean
log probability of the lowest ceil(20%×47)=10 teacher-forced token probabilities.

**Membrain score = H + log(max(A(R),1e-8)).**

This combines Min-k20% with a fixed recovery penalty. Equivalently it ranks
the hard-token geometric likelihood times recovery support. This product is a
heuristic score, NOT a probabilistic independence assertion or membership
probability. The Min-k anchor is explicit, not a hidden learned baseline.

Predeclared ablations, none eligible to replace the primary after testing:

1. Recovery only: remove H.
2. Without embeddings: set the embedding factor to 1.
3. Without rollout loss: set E=0.
4. Uniform weights: recompute the same curves with uniform suffix weights.

Report all ablations, including if the fixed correction hurts Min-k. Compare
against the six membership-label-free scores from the supplied paper: LOSS,
Min-k20%, Zlib, Lowercase, Neighbor (same documented local proposal adaptation),
and SPW-MIA. All score signs/formulas are frozen before fresh target evaluation.

## Fresh benchmark

Use public AG News **test.csv** descriptions and WikiText-2 **test.txt**
nonheading lines, rather than recycling prior benchmark/attack-development
texts. These source split names do not define membership: membership is
randomly assigned by our own target training. Normalize whitespace, require
at least 200 characters and 48 BPE tokens, and take the first 48 tokens.

Reuse only the old public-background tokenizer, 1024 background pretraining
chunks, and Neighbor proposal checkpoint. Exclude all exact 48-token chunks
from both prior benchmark data archives and all tokenizer-background documents.
Deduplicate across the two fresh corpus pools. Source files, source hashes,
chunk hashes, role assignments, and exclusions are recorded. Partial/semantic
overlap in public text cannot be ruled out by exact deduplication.

Select 384 fresh documents per corpus using fixed seed 9257. Three new target
seeds 5101/5102/5103, two corpora, two exposure regimes = twelve targets.
Same target architecture/training as before: 182,016-parameter 2-layer causal
Transformer, 64 hidden units, four heads, 1024 BPE vocabulary, tied embeddings;
background pretraining three epochs, AdamW lr .003, weight decay .01, batch32,
gradient clipping1. Within each pool/seed randomize 256 members/128 nonmembers;
evaluate first128 members and all128 nonmembers. Standard exposure four epochs;
repeated exposure twelve epochs, first32 members repeated eight times/epoch.
Both regimes share that seed's pretrained initialization. The repeated condition
changes repetition and optimization budget. Three seeds share the fresh corpus
pool and change assignments/training, not independent data populations.

The candidate scoring function never receives labels. Compute and save scores
BEFORE passing them to the metric function. SHA-256 of score code and protocol
is saved before target training; forbid resume if they change. Do not train,
load, select, normalize with, or calibrate from any previous attack classifier.

## Evaluation and interpretation

Per-target ROC AUC, conservative TPR at FPR <=1%, average precision and
trapezoidal PR AUC; equal-target macro means; 1000 paired stratified sample
bootstraps for AUC differences vs LOSS, Min-k20%, and SPW-MIA. Bootstrap
intervals condition on fitted targets, do not include model-training uncertainty,
and are not corrected for multiple comparisons. At128 negatives, FPR resolution
is .78125%, so at most one false positive is permitted below1%; no deployment
threshold or 1%FPR guarantee is inferred. Also report AP at varying prevalence
with20 shared subsamples and prefix recovery/embedding curves for all samples.

Unsupervised here means membership-label-free and no fitted attack, not that
the target LM was never trained. A threshold selected with membership labels
would reintroduce supervised calibration and is deliberately absent. Hidden
states are still needed by Membrain, stronger access than the probability-only
baselines. These CPU results cannot establish billion-parameter performance.

This design was informed by the earlier observed failures, so this is a new
held-out experiment, not an externally preregistered study. No result from the
fresh target set is used to choose among scores. The previous supervised
Membrain AUC of .9623 is not the performance claim for this label-free method.

## Run

Prepare the previous public tokenizer/background/proposal artifacts, then:

```sh
python -m unittest discover -s experiments -p test_membrain_unsupervised.py
python experiments/run_membrain_unsupervised.py --download-data
python experiments/report_membrain_unsupervised.py
```

Outputs: `results/membrain_unsupervised`. Use a fresh output directory if code
or protocol changes. Historical supervised scripts and outputs are preserved
for audit, clearly labeled as supervised.
