# Membrain: SPW-inspired, membership-supervised continuation trajectories

**Historical supervised experiment.** The current Membrain attack is the
label-free fixed score in [MEMBRAIN_UNSUPERVISED_PROTOCOL.md](MEMBRAIN_UNSUPERVISED_PROTOCOL.md).
The supervised results here must not be attributed to that unsupervised method.

This extension retains the name **Membrain**. The old origin-CNN gap is retained
as `Membrain original` in comparisons, not silently replaced in historical runs.
This protocol was written before evaluating the fresh target benchmark.

## What changes and what stays

Membrain still supplies increasing TRUE prefixes and generates the WHOLE suffix
greedily without teacher forcing. It compares hidden states of true and generated
continuations. Three changes address the earlier failure:

1. Compare hidden states at aligned suffix positions before averaging. Retain
   the pooled embedding trajectory as well, so local recovery is not lost.
2. Use SPW-MIA's adaptive shortening to identify context-sensitive positions.
   For a suffix position t, let `g_t = log p_full(true_t) - log p_short(true_t)`;
   weights are `0.5 / suffix_length + 0.5 * softmax(clip(g,-20,20))` within each
   suffix. Weight tokenwise embedding cosine, recovery accuracy, and free-running
   CE against TRUE tokens. This is an SPW-inspired weighting choice, not the
   paper's exact SPW scalar. The actual SPW scalar/components are also inputs.
3. Train a membership classifier on independent shadow models with known member
   and nonmember documents. Do not train only on trajectory origin. A regularized
   logistic classifier consumes the complete curves and their area/change/min/
   max summaries, plus LOSS, Min-k20%, token likelihood summaries, and SPW
   statistics. Thus any gain must be tested against a probability-only learner
   with identical supervision, not just against fixed unsupervised baselines.

Curves: pooled, aligned, weighted embedding cosine; suffix token and exact
recovery; free-running true-token CE, its excess over teacher-forced CE, weighted
CE, weighted accuracy; and teacher-forced suffix CE. Prefix fractions are the
same ten points from .05 to .9 as before. There is no generated self-likelihood
substitution. Ground-truth suffixes are targets, never free-running inputs.

## Data and training, fixed before target evaluation

Reuse only the old public-background tokenizer, public-background training
chunks, and neighbor proposal checkpoint. All new attack-development and target
documents exclude ALL old experiment chunks and all tokenizer-background chunks.
Use the next 2496 unique 48-token chunks per corpus after the old preprocessing.
Exact chunk deduplication spans both corpora and roles; semantic overlap remains
possible. This is a CPU-scale test, not a billion-parameter paper reproduction.

Roles per corpus, disjoint by document:

- Three 384-document shadow fitting pools, seeds 3101/3102/3103.
- One 384-document shadow validation pool, seed 3201.
- One 384-document shadow probability-calibration pool, seed 3301.
- One 384-document fresh target pool, target seeds 4101/4102/4103.
- 192 auxiliary origin documents used only to rerun the original origin CNN.

Each model has exactly the previous architecture, 1024 background chunks and
three background pretraining epochs. Each 384-document pool is randomly assigned
256 members and 128 nonmembers. Evaluate 128 members (including the first 32
heavily repeated ones) and all 128 nonmembers. Ordinary training = four epochs;
repeated training = twelve epochs, first 32 members repeated eight times per
epoch. Architecture, optimizer and all other target settings match the previous
suite. Train both regimes in both corpora: 12 fitting shadow models, four
validation models, four calibration models, and twelve fresh target models.
Targets share a fresh document pool across seeds but never across development
roles. Different seeds reassign membership; seeds are not new independent data.

Train-only feature standardization (minimum std .001), clip standardized inputs
to [-8,8]. Logistic objective = mean binary CE + lambda*||w||²/2; fit using
LBFGS. Select lambda from [.001,.01,.1,1] by macro validation-target ROC AUC,
prefer stronger regularization on ties. Refit on fitting + validation shadows,
then positive-slope Platt calibrate on separate calibration shadows. Freeze
selection, weights, calibration, and thresholds BEFORE training/evaluating any
fresh target. Never use fresh target statistics for normalization or labels
for fitting, tuning, calibration, threshold selection, or score orientation.

Primary method = all features, called Membrain. Predeclared controls use the
same selection procedure and training/calibration labels:

- Probability-only learner: likelihood summaries, teacher-forced suffix CE,
  SPW features, no generated continuation or hidden-state comparison.
- Membrain without embeddings: also includes generated recovery/CE curves.
- Membrain curves only: embedding/generation curves, no probability-only group.

Rerun the original Membrain origin CNN, LOSS, Min-k20%, Zlib, Lowercase,
Neighbor (same local masked-LM adaptation), and SPW-MIA on all twelve fresh
targets. No post-hoc winner becomes the primary Membrain.

## Evaluation

Macro ROC AUC, conservative TPR at <=1% FPR, average precision, 1000 stratified
paired bootstrap intervals per target, with differences versus old Membrain,
LOSS, Min-k20%, SPW-MIA, and the probability-only learner. Low-FPR resolution is
still 1/128 = .78125%; report uncertainty. All sample intervals condition on a
fitted target. Report all seeds/corpora/regimes and same-label supervision
ablations, including failures. Repeat imbalance AP tests using 20 shared draws.

Also report held-out Brier/log loss and the realized TPR/FPR at a FIXED threshold
chosen for <=1% FPR on pooled calibration shadows. This threshold has no guarantee
on fresh targets. Membership probabilities assume the balanced shadow prior and
matched training distribution; they are not calibrated for arbitrary external
texts or priors. Membrain now requires labeled shadow training and hidden-state
access, stronger assumptions and extra compute than the fixed-score baselines.

The previous benchmark motivated this revision, so the old test set is treated
as development history. Fresh targets protect this run from direct reuse of
those test labels; this is not a preregistered external replication.

## Run

After preparing the previous suite's tokenizer/background/proposal/data:

```sh
python -m unittest discover -s experiments -p test_membrain_prefix.py
python experiments/run_membrain_prefix.py --stage development
python experiments/run_membrain_prefix.py --stage test
python experiments/report_membrain_prefix.py
```

Default output: `results/membrain_prefix`. `--stage all` executes both stages
in order. Completed artifacts resume; use a fresh output after changing code
or protocol. The frozen classifier is a trusted local PyTorch artifact, not an
untrusted checkpoint loader. Do not load externally supplied pickle checkpoints.
