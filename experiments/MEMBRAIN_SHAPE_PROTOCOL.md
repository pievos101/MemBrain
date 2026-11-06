# Membrain shape experiment: geometry and hard-token pooling

Research experiment, not an automatic promotion of a new best method. All rules
below are fixed before target training. Membership labels enter only target
construction/training and metric evaluation, never scoring, fitting, selection,
normalization across candidates, calibration, or threshold choice. No origin or
membership classifier is trained or loaded. Scores are not probabilities.

## Hypotheses

1. A constant origin offset can obscure agreement of trajectory SHAPE. Preserve
   one suffix vector per prefix and origin, normalize each vector, center each
   candidate trajectory over its own ten prefixes, and compare shapes without
   an arbitrary rotation or alignment fit.
2. Generic easy tokens may dominate pooled embeddings. Form an additional pair
   of trajectories by mean-pooling ONLY the hardest ceil(20% * suffix length)
   positions according to teacher-forced true-token CE. Use the SAME chosen
   true-token positions for true and generated embeddings. No generated-token
   self-likelihood is substituted for loss against the true tokens.

For each pair of trajectories, compute:

- Proximity: normalized early-weighted area of pointwise cosine.
- Centered correlation: Frobenius cosine between the centered trajectories.
- Velocity alignment: early-weighted area of cosine between adjacent trajectory
  steps; use midpoints of the prefix grid. Zero-length vectors give zero evidence.
- Shape score: fixed equal mean of proximity, centered correlation, and velocity.
- Diagnostic path-length agreement: -abs(log((length_true+1e-6)/(length_gen+1e-6))).

Normalize vectors BEFORE centering. Prefix grid [.05,.1,.2,...,.9] and area
density(1-r) match the preceding experiment. Shape scores lie in[-1,1].

Predeclared primary candidate: **Membrain shape hybrid = Min-k20% + 0.5 * hard-token
shape score**. This is explicitly a Min-k anchor plus bounded geometry correction,
not evidence that high performance comes from geometry. The coefficient0.5 is
fixed in advance, not fitted. Do not replace it with a best test-set variant.

Report both geometry-only scores; all individual geometric components for both
pooling methods; ordinary and hard-token teacher/rollout CE curves; and the
Min-k anchor alone. In particular, the hard-token teacher-CE curve tests whether
any improvement is explained by likelihood rather than embeddings. Also rerun
all six paper scores and the prior SPW-weighted fixed Membrain on the same target
texts. Earlier failed experiments motivate these hypotheses and are development
history, not external preregistration.

## Fresh targets

Use next384 unique chunks per corpus from the same AG News/WikiText-2 test-source
files and fixed9257 shuffle. Exclude ALL exact chunks from all previous studies
(including the simple two-curve study) and ALL tokenizer background documents.
Reuse public-background tokenizer/training chunks/Neighbor proposal only. Exact
deduplication cannot exclude semantic/partial overlap.

Twelve targets: seeds7101/7102/7103 × two corpora × two exposure conditions.
Same tiny182,016-parameter causal Transformer,48 tokens,3 background pretraining
epochs;256 target members/128 nonmembers, evaluate128+128. Ordinary4 epochs;
repeated12 epochs with first32 members repeated8 times/epoch. Same AdamW .003,
batch32,weight decay .01, gradient clipping1. Seed pools are shared, not twelve
independent datasets. Exposure conditions also change total optimization budget.

Freeze scoring/runner/protocol hashes before training. Save candidate-only scores
before metric computation. Report per-target and macro ROC AUC, empirical
TPR<=1%FPR, AP, and1000 paired stratified sample-bootstrap intervals vs LOSS,
Min-k, SPW and hard-token teacher-loss curve. Sample intervals condition on fitted
models, omit training uncertainty and multiplicity correction. At128 negatives,
1%FPR allows one false positive. No deployment threshold is fitted.

Report all variants even if no geometry score adds value. No promotion of the
best test result as a newly validated primary. These are CPU research benchmarks,
not large-model replication or validated external membership probabilities.

Run `python experiments/run_membrain_shape.py`, then
`python experiments/report_membrain_shape.py`. Default output:
`results/membrain_shape`. Prior public source/chunk archives are needed for exact
exclusion. A fresh output directory is required after any frozen-code change.
