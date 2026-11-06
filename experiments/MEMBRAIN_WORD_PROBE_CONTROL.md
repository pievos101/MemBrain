# Exploratory output-only next-word control

Designed after the main output-only study's results, so this is exploratory on
the same models/texts, not independent held-out confirmation. No main study
score/protocol is changed or replaced. Freeze this control's source/protocol
before collecting the new short completions.

Hypothesis: whole-suffix failure compounds prediction errors; ordinary members
may still show higher conditional fit on the FIRST missing word. Query the
same original increasing prefixes (no reversed-prefix queries) for16 IID
temperature0.8 completions capped at8 new tokens, plus one greedy completion.
The scorer sees only strings. Compare the first lexical word to the known
candidate suffix's first lexical word. Generation cap can censor long words;
this is a text-event frequency, not the model's true token probability.

At each prefix count h correct samples out of16; use fixed Jeffreys smoothing
p=(h+0.5)/17. Primary exploratory score: early-weighted integral of log(p) on
the actual prefix grid. Other fixed controls: mean of the lowest ceil(20% of
prefix-count) log(p), greedy first-word recovery area, sampled first-word
recovery area. No coefficients, score signs, or sample counts are chosen by
membership labels. Candidate text supplies self-supervised word targets; no
membership/origin classifier is trained.

Fit one-vs-two component mixtures to only the same128 unlabeled discovery
scores using the preceding fixed GMM procedure. Orient by higher score and
abstain if one component is preferred; evaluate the other128 texts only after
the unsupervised model is saved. A component posterior is not membership
probability. Prior use of these evaluation texts limits confirmation strength.

Report all scores and grouping, including failures, and query costs. Do not
promote a test-set best alternative as a newly validated primary. This is an
information-access experiment: a word-recovery frequency estimated from text
samples is allowed, unlike reading the model's probabilities. All targets are
the earlier tiny local models, not externally hosted LLMs.
