# Membrain: two embedding curves plus absolute loss

Current experiment, fixed before the new benchmark. The user wants one embedding
for the true suffix and one for the generated suffix at each increasing prefix.
Keep those two vectors explicitly; simplify away adaptive token weighting, shifted
nulls, Min-k anchoring, and the rollout-minus-teacher loss gap from primary Membrain.
No attack classifier, membership labels, shadow training, calibration, batch
normalization, or fitted coefficients. Outputs are scores, not probabilities.

For each of the ten prefix fractions .05,.1,.2,...,.9, greedily generate the WHOLE
remaining 48-token sample suffix. Mean-pool the model's final hidden states AFTER
the suffix tokens, separately for the true and generated suffix. Prefix tokens
are not included in the pool. Store both vectors, not just their distance.

Define `d(r)=(1-cos(z_true(r),z_generated(r)))/2`, clipped to [0,1]. Let `T(r)`
be mean teacher-forced CE against the true suffix and `G(r)` be mean CE against
TRUE suffix tokens while the model conditions on its own generated tokens.
This is not generated-token self-likelihood. No true suffix tokens are fed into
the free-running branch.

**Membrain = -A[T(r) + d(r)*G(r)]**, where A is normalized trapezoidal integration
over [.05,.9] with fixed density (1-r). Higher means stronger membership evidence.
Both terms have nats/token units: confident true continuation prediction, plus
an absolute rollout-loss penalty gated by embedding mismatch. No claim that this
heuristic is a membership probability or an optimal combination is made.

This directly addresses the earlier failure: subtracting teacher loss from
rollout loss may penalize members merely because teacher loss improves faster
than greedy recovery. Absolute loss avoids that specific subtraction. Embedding
similarity alone may reflect generic fluency; teacher loss helps distinguish
plausible continuation from confidence in the specific supplied text.

Predeclared diagnostics: embedding curve alone (-A[d]), teacher loss curve alone
(-A[T]), rollout loss curve alone (-A[G]), no rollout weighting (-A[T+d]), and
loss curves without embeddings (-A[(T+G)/2]). None replaces the primary after
seeing results. Recompute previous fixed Membrain and all six paper baselines
on the same new targets. Explicitly compare to the teacher-curve control and
Min-k20%, not merely to the much weaker old origin classifier.

The true/generated paths can be called origin classes 0/1; these labels are
automatically available. Training a classifier on them is self-supervised origin
discrimination, not membership training. Its output cannot become a membership
probability without an additional justified link. This experiment measures their
paired separation directly, keeping the attack entirely unfitted. Correctly
remembered text need not have been trained on, and membership need not cause
exact greedy reproduction. The hypotheses must survive benchmarking.

## Fresh evaluation

Take the next 384 unique chunks per corpus from the public AG News test source
and WikiText-2 test source with previous normalization/200-character/48-token
rules and shuffle9257. Exclude ALL chunks from the three preceding experiments
and ALL tokenizer-background documents, deduplicating across corpora. Public
background tokenizer/1024 pretraining chunks/Neighbor proposal may be reused;
no previous attack classifier is loaded. Exact exclusion is not semantic
deduplication. Prior experiments inform the hypothesis, so this is not an
externally preregistered study.

Twelve targets: seeds6101/6102/6103 × two corpora × ordinary/repeated exposure.
Same 182,016-parameter causal Transformer and optimization as the preceding
suite: background3 epochs,256 members,128 nonmembers; evaluate128 members and
all128 nonmembers. Ordinary4 epochs; repeated12 epochs with first32 members
repeated8 times/epoch. Seeds share corpus pools but vary assignments/training.
Repetition also changes optimization budget. These are small CPU experiments,
not the supplied paper's billion-parameter reproduction.

Freeze hashes of scoring, runner, and this protocol BEFORE target training.
Candidate scoring receives no labels. Save candidate-only scores before metric
computation. Report all targets, all controls, paired1000-draw sample bootstrap
AUC intervals versus LOSS/Min-k20%/SPW-MIA and the teacher-curve control, conservative
empirical TPR<=1%FPR, and AP. At128 negatives, only one false positive is allowed
below1%; no operational threshold is fit. Bootstrap conditions on fitted targets;
it omits training uncertainty and multiple-comparison correction. No raw score
pooling across different target models. Additional fixed diagnostic scores may
be plotted, but may not be promoted as a new primary after seeing the data.

Plots must include all-sample prefix curves and saved true/generated embeddings.
Any displayed two-dimensional projection uses shared per-target PCA, fit without
membership labels for visualization only; never use projected coordinates in
the score. Use predetermined example rows, never examples selected for success.

Run `python experiments/run_membrain_simple.py`, then
`python experiments/report_membrain_simple.py`. Requires the previous suites'
public source and chunk archives for exact exclusion. Results go to
`results/membrain_simple`. Use a fresh directory if the score/protocol changes.
