# Prospective validation of output-only Membrain next-word recovery

The prior16-sample next-word score (.7106 macro AUC) was exploratory. This new
study freezes that EXACT score and generation settings before fresh target
training. It is a prospective internal validation, not external preregistration.
No membership labels enter scoring, discovery fitting, orientation, thresholds,
model/sample-count selection, or calibration. All variants are predeclared;
none replaces the primary based on this test.

## Primary and controls

Use unchanged `membrain_word_probe.score_record`: at each existing prefix grid
count correct first lexical words in16 IID temperature0.8 completion strings,
cap8 new tokens, Jeffreys smoothing(h+.5)/17, and early-weighted log-frequency
area. Prefixes/word normalization are unchanged from `membrain_text.query_plan`.
No model internals or tokenizer IDs enter these scoring functions. Candidate
text provides self-supervised next-word targets, not membership labels.

Primary = **Membrain16** from seed(target_seed+9517). An independent16-sample
query replicate uses seed(target_seed+19517). Predeclared secondary scores:
32 pooled samples, first3 nested samples of the primary batch (same horizon,
budget control), hard-prefix log-frequency, greedy next-word recovery, greedy
whole-suffix/fixed-tail edit recovery (separate cap96 queries), and negative
reference word count as a metadata-only control. No extra long stochastic or
reversed-prefix queries are needed for this validation.

Estimate query stability with tied-rank Spearman correlation between the two
16-draw runs, replicate AUC differences, and high-group Jaccard on held-out texts
when both mixtures select two components. Do not replace primary results with
the replicate or pooled32 result. More samples change query budget; report costs.
The short cap may censor unusually long first words. These are empirical text
event frequencies, not exact model word/token probabilities.

For each of primary16, replicate16 and pooled32 fit the unchanged one-vs-two GMM
BIC procedure on128 UNLABELED discovery scores, orient higher mean as high
recovery, and abstain if one component is preferred. Evaluate128 different
sentences. Randomize candidate order BEFORE querying, then use the fixed8271
unlabeled discovery permutation. No stratification, class priors, or labels enter
the fit. Component posterior is not calibrated membership probability. Report
actual held-out prevalence and all abstentions, not just successful targets.

## Independent data and capacity checks

Use AG News TRAIN descriptions and WikiText-2 VALIDATION paragraphs as fresh
source documents. Source split names do not define membership, which is assigned
by our own target training. Retain prior sentence eligibility: one sentence per
document,12–40 lexical words,16–80 BPE tokens, sentence-ending punctuation;
normalize whitespace. Select768 per corpus with shuffle44291, forming two
DISJOINT384-document pools. Exact-exclude all prior studies' document chunks,
ALL tokenizer-background documents, and all prior full candidate sentences.
Deduplicate across new pools/corpora. Partial/semantic overlap remains possible.

Targets: two architectures × two corpora × two exposure regimes × two seeds =
16 targets. Seed9101 uses pool0,9102 pool1; capacities/regimes share the same
seed's pool and membership assignment to permit matched comparisons. These
16 models are not16 independent datasets. Do not pool incompatible raw scores
across models. Macro means weight the planned cells equally; show all strata.

Small model: previous2-layer,64-dimensional,4-head causal Transformer.
Larger model:3 layers,96 dimensions,4 heads. Same1024-token public-background
BPE tokenizer and max context256, tied embeddings, learned positions. Train
each seed/capacity background model for3 epochs on the same public1024 chunks;
no prior target checkpoint is reused. Per target256 members/128 nonmembers;
evaluate128 members and128 nonmembers. Standard4 epochs. Repeated12 epochs,
first32 members repeated8 times/epoch (96 presentations), others12. Same
AdamW .003,weight decay .01,batch32,clip1,variable-length padded sentence
training with ignored PAD labels and newline termination. Repetition also
changes optimization budget. Randomize the evaluation order independently
with seed(target_seed+11417); retain labels only in owner-side evaluation.

Heavy exposure is a training-frequency label, not an oracle for internal
memorization. These small local models emulate text APIs; no commercial API
or billion-parameter claim follows, including for the larger local capacity.

## Evaluation and access separation

Save all raw string transcripts, label-free scores, mixtures, and decisions
BEFORE owner-side labels and privileged LOSS/Min-k/SPW evaluation. Privileged
references are not output-only competitors and never enter Membrain scoring.
Report raw membership AUC/AP/conservative TPR<=1%FPR, heavy-exposure AUC in
repeated strata, held-out group precision/recall/balanced accuracy/FPR, query
stability, all capacities/pools/regimes, and all query costs. Use1000 paired
stratified sample bootstraps for per-target AUC intervals/differences, including
versus the greedy text-only whole-suffix control. Those intervals condition
on fixed targets/queries, omit training/query uncertainty and multiplicity
correction; independent query replicates partially assess sampling sensitivity.
At128 nonmembers,1%FPR permits one false positive; no deployment guarantee.

Freeze hashes of old score/query modules, new validation helpers/runner, and
this protocol before target training. No score tuning or promotion after
evaluation. Compare the new confirmation point estimates with development
descriptively; texts/source partitions and model matrix differ, so the older
.7106 is not a paired baseline. A favorable repeat-exposure result alone is
insufficient to claim reliable ordinary training membership detection.

Run `python experiments/run_membrain_text_validation.py`, then
`python experiments/report_membrain_text_validation.py`. Default output:
`results/membrain_text_validation`. Requires public raw source/chunk archives
from preceding studies. Use a fresh output after frozen-code changes.
