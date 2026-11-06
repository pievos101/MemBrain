# Membrain: output-only recovery distributions and unsupervised grouping

Fixed before the new targets. End goal remains UNSUPERVISED or SELF-SUPERVISED
membership/memorization detection. This experiment uses unsupervised mixture
fitting, with NO membership labels in scoring, fitting, model choice, component
orientation, calibration, or thresholds. Benchmark labels are revealed only for
retrospective evaluation. No supervised membership/origin classifier is used.

## Access boundary

`membrain_text.py` receives candidate text and a completion callback. The callback
receives ONLY prefix strings and generation settings. It returns greedy and
sampled TEXT strings, never hidden states, probabilities, losses, tokenizer IDs,
gradients, or weights. An opaque local service implements generation for this
CPU benchmark, as a hosted API would. Its private sampling logits are not attack
inputs. No commercial hosted model is queried. Raw unlabeled JSONL transcripts
make scoring independently reproducible from strings alone.

The benchmark owner may train models and compute privileged LOSS/Min-k/SPW
references separately AFTER text-only scoring and unsupervised fitting. Clearly
label those as stronger-access references, not output-only competitors.

## Sentences and query plan

Use complete natural sentences, NOT arbitrary fixed BPE chunks. Select one
eligible12–40 lexical-word sentence (16–80 BPE tokens) per fresh source document,
ending in sentence punctuation. Source preprocessing is the preceding public
AG News/WikiText-2 test-source normalization. Simple punctuation segmentation
and WikiText artifacts remain limitations. Exact-exclude every previous study's
source-document first48-token chunks, all tokenizer-background documents, and
duplicate sentences across pools. Semantic overlap cannot be ruled out.

At eight requested stages [.05,.1,.2,.35,.5,.65,.8,1], give round(stage*(W-4))
lexical words, clipped to[1,W-4], deduplicating repeated prefix lengths. Store
actual supplied-word fraction p/W; the final four reference words always remain
unseen. Query completions until newline termination or a fixed96 new-token cap,
never supplying the reference suffix or its token length to the service.

At every prefix get THREE independent temperature0.8 samples and ONE greedy
completion. The greedy output is evaluated separately, never pooled as an IID
sample. Three samples give only coarse empirical distributions; no precise
generation-probability estimate is claimed. Also query the word-reversed prefix
with the same sampling settings as a diagnostic control; at one word it is
identical. All prompts/settings/counts are fixed in advance, not selected on
membership evaluation. Calls are local but query budgets are recorded.

## Text-only distances and fixed scores

Normalize Unicode NFC, casefold, and extract lexical words including internal
apostrophes/hyphens, ignoring punctuation. Word-edit similarity is one minus
Levenshtein distance divided by the longer word-sequence length. Also report
normalized character edit (on joined words), ROUGE-L F1, and exact word match.
No embedding model or learned semantic metric is used.

At each prefix retain the empirical distance distribution from the three IID
outputs. Different prefixes are DIFFERENT conditions, not IID observations.
For distances in[0,1], Wasserstein-1 distance to a zero-distance point mass is
their mean. Thus primary **Membrain text** = early-weighted integral of
`1-mean(fixed-tail word-edit distance)` over the actual p/W grid, normalized by
the same integral of density(1-p/W). Compare the SAME final four lexical words
of reference and completion to avoid changing reference targets. Full-suffix
scores remain explicit comparisons; generation length/termination can affect
tail extraction and are limitations.

Predeclared alternatives: sampled whole-suffix edit; greedy whole/fixed-tail
edit; sampled tail character edit, ROUGE-L and exact recovery;25th-percentile
tail recovery; sustained recovery speed (earliest prefix with mean similarity
>=.75 at that and every later prefix, else0); single longest-prefix recovery;
and true-prefix minus reversed-prefix recovery area. No alternative becomes the
primary after evaluation. Primary score is per sentence, label-free, not a
membership probability.

## Unsupervised two-class attempt

For each target, randomly split its256 evaluated texts into128 unlabeled
discovery and128 held-out texts with seed8271, WITHOUT stratifying by labels.
Fit one-dimensional Gaussian mixtures to ONLY the discovery primary scores:
one vs two components, BIC parameter counts2/5, variance floor1e-4, mixture
weights clipped to[.05,.95], three deterministic quantile initializations,
up to200 EM iterations. Select highest unlabeled likelihood initialization.
The component with greater mean recovery is always the memorization-LIKE
component; never orient using true membership labels. If BIC favors one
component, abstain from assigning two classes. Otherwise classify held-out
sentences at high-recovery-component posterior>=.5.

That posterior is a COMPONENT probability, not calibrated membership probability.
Gaussian fit/BIC is a heuristic model comparison on bounded, sometimes discrete
scores, not proof of natural bimodality. Discovery/test separation evaluates
out-of-sample grouping. Report any abstention, selected component sizes, held-out
membership precision/recall/balanced accuracy, and heavy-exposure detection in
repeated conditions. Heavy exposure is known training frequency, NOT an oracle
label proving a sentence is memorized. Membership and verbatim memorization are
different targets; useful recovery clusters need not recover all members.

## Target training and evaluation

Twelve new sentence-trained targets: seeds8101/8102/8103 × two corpora × two
exposure regimes.384 fresh documents per corpus; randomly assign256 members and
128 nonmembers, evaluate first128 members and all128 nonmembers. Ordinary4
epochs; repeated12 epochs with first32 member sentences repeated8 times/epoch
(96 vs12 presentations). Seeds share corpus pools but change assignment/training.
Train the same182,016-parameter causal Transformer with existing public1024-chunk
background3-epoch pretraining, AdamW .003/weight decay.01/batch32/clip1. Pad
variable sentence lengths with ignored PAD targets, and append one newline
terminator token. Score privileged references without newline or padding.

Report all12 targets, macro ROC AUC/AP/conservative TPR<=1%FPR for raw scores,
1000 paired stratified sample-bootstrap intervals vs privileged LOSS/Min-k/SPW,
and held-out unsupervised grouping separately. At128 nonmembers,1%FPR permits
one false positive. Sample bootstraps condition on trained targets, do not
include target-training uncertainty, and are not multiplicity-adjusted.
Score/query/mixture code and protocol are hashed before target training.
Natural sentences, new training and termination change this benchmark, so do
not claim paired gains against previous studies' numerical values. No external
large-model or API performance claim follows from tiny-model CPU tests.

Run `python experiments/run_membrain_text.py`, then
`python experiments/report_membrain_text.py`. Raw transcripts, scores, unlabeled
mixtures and reports go under `results/membrain_text`. Need `rapidfuzz==3.9.7`
and preceding source/chunk archives for exact exclusions. Prior studies motivate
this new held-out experiment; it is not external preregistration.
