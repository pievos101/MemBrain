# Membrain: cluster full recovery curves instead of scalar scores

Exploratory reanalysis using saved text-only completion transcripts. No new model
queries or training are required. These data were evaluated previously; the
discovery/held-out split is respected for clustering, but this is NOT independent
fresh validation. Freeze this protocol and scoring/fitting/runner source before
running the new comparison. No settings or method are selected by membership
labels. No best test result is promoted as a new validated primary.

## Inputs and preserved information

Use all16 prospective-validation targets (two capacities, two corpora, two
exposure regimes, two independent pools). Read only their saved primary16
short-completion strings and greedy whole-suffix strings to construct features.
At every supplied prefix compute normalized WORD Levenshtein distance between
generated whole suffix and original suffix. Retain the full curve. Separately
retain the empirical next-word mismatch rate from16 IID returned strings.
For Bernoulli word match, this rate describes the full two-point empirical
distribution at that prefix. No model probabilities or tokenizer are used.

Align each curve to eight fixed points uniformly spaced over[.125,1] of the
available prefix (ending leaves four words). The input query grids already
cover that common interval. Interpolate only between queried prefix points;
no extrapolated early predictions. This aligns varying sentence lengths. The
suffix-composition confound remains for whole-suffix distance and is disclosed.

Fit on eight-dimensional whole-suffix distance vectors, eight-dimensional
next-word mismatch vectors, and sixteen-dimensional concatenated vectors.
Do NOT average these vectors before fitting. Coordinates are raw distances
in[0,1], with no label-based or target-normalized scaling/feature selection.

Matched scalar controls use exactly those resampled curves: early-weighted
area (density1-r) and simple mean, for each signal. Use the SAME mixture family,
variance floor, fitting starts and discovery split as for vector methods. Also
report raw scalar recovery ranks separately. This isolates compression effects
from changes in sampling/grid/scoring. The archived16-sample log-frequency
score is contextual comparison only; its transformation/grid differ.

## Label-free clustering

Primary comparison: diagonal Gaussian mixtures selecting one vs two components
by BIC on128 UNLABELED discovery vectors. Use four deterministic farthest-first
starts (seeds3/17/29/53), max200 EM iterations, tolerance1e-7, coordinate variance
floor.001, minimum component mass2, weight floor.05. Parameter count for k
components,d dimensions =2dk+k-1. Select highest unlabeled likelihood start;
select lowest BIC k, tie favoring fewer components. Diagonal covariance is a
regularized approximation: correlated prefix coordinates can distort BIC.

Order fitted components by early-weighted distance of their prototype curve,
lowest distance first. This scalar is used ONLY for deterministic group naming
after vector fitting, never as a substitute for the vector input. For two groups,
report the posterior of the highest-recovery component and assignment>=.5. If
one group is selected, abstain; use constant.5 for an uninformative ROC score.
Posterior is COMPONENT probability, not membership probability. Low recovery
does not prove nonmembership. No class is flipped using evaluation labels.

Predeclared sensitivity: whole-curve fit selecting one/two/three components by
BIC. With three groups, highest-recovery group remains the same fixed definition;
do not merge groups using labels. Other groups may represent intermediate
recovery, not membership classes. Do not promote a best k after testing.

## Evaluation

Reuse exactly each target's128 discovery/128 held-out index split. Save every
feature matrix, fitted model, selected k and held-out score BEFORE loading
benchmark membership labels. Evaluate only held-out sentences: ROC AUC/AP,
group precision/recall/balanced accuracy/FPR, component-selection counts, and
high-exposure ranking in repeated conditions. Report every capacity/regime.

Use1000 paired stratified sample bootstraps for vector-versus-matched scalar
posterior AUC differences. These condition on each fitted model/query cache,
omit fitting/training/query uncertainty and multiplicity correction, and share
documents across capacity/regime targets. They are exploratory diagnostics.
Every prior failure and alternative is retained; no membership labels enter
feature construction, covariance fitting, normalization, k selection, naming,
or thresholds. True labels enter only retrospective evaluation.

Plot actual full profiles/prototypes and held-out comparisons. Label diagonal
GMM assumptions, possible overfitting at d16/n128, abstentions and data reuse.
These tiny local models do not establish closed-provider or large-LM performance.

Run `python experiments/run_membrain_curve_cluster.py`, then
`python experiments/report_membrain_curve_cluster.py`. Default output:
`results/membrain_curve_cluster`. Cached transcripts from the preceding validation
are required. Use a fresh output directory if frozen source/protocol changes.
