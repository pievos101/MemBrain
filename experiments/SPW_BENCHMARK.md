# SPW-MIA and Membrain: independent small-scale benchmarks

This is an independent implementation of the methods in Xie et al.,
*Shortening the prefix! Members and non-members exhibit divergent behavior*,
Knowledge-Based Systems 346 (2026), 116100, DOI: 10.1016/j.knosys.2026.116100.
The supplied PDF is the specification. This experiment does **not** reproduce
the paper's model/dataset scale or its reported numerical results.

## Protocol declared before execution

Eight cells: two natural-text corpora (AG News descriptions and WikiText-2
nonheading paragraphs), two target training seeds (101,202), and two exposure
regimes. Each target is a 2-layer causal Transformer, hidden dimension 64,
four attention heads, tied embeddings, no dropout, learned positions, and
1024 byte-BPE tokens. The three special tokens cannot be output. Samples are
the first 48 BPE tokens of a document. CPU training uses AdamW, learning rate
0.003, weight decay 0.01, batch 32, gradient norm clipped at 1.

The tokenizer is trained exclusively on 2048 reserved documents per corpus.
1024 chunks (512 per corpus) from that public background pretrain each causal
target for three epochs. A separate bidirectional masked Transformer trains on
the same background for four epochs and supplies neighbor proposals. Neither
background model sees target member, nonmember, or auxiliary documents.

Whitespace is normalized; documents need at least 300 characters. Exact first-
chunk duplicates are removed across background and all experiment roles. Each
corpus supplies a fixed 384-document membership pool and 192 auxiliary
documents. Within each seed, a random permutation assigns 256 pool documents
to target training and 128 to nonmembership. Evaluation uses the first 128
training documents and all 128 nonmembers. Seeds reassign pool membership;
they are training replicates, not independent new corpora. Document-level
deduplication cannot exclude all semantic or partial overlap in public text.

Standard exposure: four epochs over 256 members. Repeated exposure: twelve
epochs, with the first 32 members occurring eight times per epoch (96 total
presentations), and the other members twelve times. Both regimes initialize
from the same public-background checkpoint for each seed. No score orientation,
hyperparameter, or epoch count is selected on membership evaluation results.
The two exposure conditions change both exposure and total optimization steps;
they are stress tests, not an isolated causal estimate of duplication effects.

## Implemented attacks

- **LOSS:** mean next-token log probability, excluding only the first token.
- **Min-k20%:** mean log probability of the lowest ceil(0.2 * token count).
- **Zlib:** negative perplexity divided by Python zlib compressed UTF-8 bytes.
- **Lowercase:** lowercase perplexity minus original perplexity. Lowercasing
  retokenizes the decoded original sample; all resulting tokens are scored.
  These perplexity formulas follow this paper rather than variants using loss
  ratios or log-perplexity normalization.
- **Neighbor:** original NLL subtracted from mean NLL of ten neighbors.
  Each neighbor replaces one token at a distinct seeded random position using
  the highest-probability nonoriginal output of the background masked model.
  This is a **local proposal adaptation**: the paper does not fully specify its
  neighbor candidate generation. It is not the original pretrained proposal
  model or an exact replication of the cited Neighbor implementation.
- **SPW-MIA:** equations 1–9 and Algorithm 1. For each target token from the
  third onward, test the last i tokens of its original prefix, starting at i=1.
  Stop at first exact equality of normalized character edit similarity to the
  initial greedy token, including zero-equals-zero. Accumulate differences
  before that stopping point only; divide by original prefix length. KL is
  between normalized **position-wise** probability vectors, not vocabulary
  distributions. The score is weighted similarity minus KL minus weighted
  maximum probability. Full-prefix evaluation guarantees termination. Decoding
  single byte-BPE tokens can produce replacement characters; this is a stated
  token-to-string convention. Cached/batched inference changes no scores.
- **Membrain:** for increasing true prefix fractions
  [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], greedily generate the entire
  remaining suffix without teacher forcing. At each prefix, average final
  hidden states **after** the suffix tokens for true and generated sequences.
  A small CNN learns true trajectory = class 0, generated trajectory = class 1
  using only disjoint auxiliary documents: 96 training, 32 early stopping,
  32 temperature calibration, 32 held-out origin evaluation. Train up to 80
  epochs, patience 12, AdamW lr .001, weight decay .01. Train/evaluation both use
  greedy generation. The declared membership score is
  `-abs(q(true trajectory) - q(generated trajectory))`, higher = more converged.
  True-origin probability, signed gap, and direct cosine similarity are
  secondary diagnostics, not candidates for post-hoc primary selection.

The classifier estimates **trajectory origin**, not membership probability.
Membership probability calibration would require a separate labeled membership
calibration set with a justified prior. No such probability claim is made.
Membrain uses model hidden states and auxiliary origin training, a stronger
access assumption than the other six attacks. Compute costs also differ.

## Evaluation and scope

Report tie-aware ROC AUC; conservative empirical TPR at FPR <= 1%; average
precision and trapezoidal PR AUC separately. With 128 negatives, the FPR step
is 0.78125%, so only one false positive is allowed below 1%; low-tail results
have substantial sampling uncertainty. Interpolated TPR is saved separately.
1000 stratified paired sample bootstraps provide per-cell AUC intervals and
paired differences from LOSS. These condition on one fitted target; they do
not substitute for independent training seeds. Macro averages give cells equal
weight; do not pool incompatible raw attack score scales.

Also evaluate seven SPW component combinations and imbalanced membership:
ratios .05,.1,.2,.4,.8,1 with all 128 negatives, rounded positive counts, and
20 shared seeded subsamples per ratio. Chance AP equals the actual sampled
positive prevalence. Plot full recovery trajectories, not selected examples.

This suite does not include the paper's nine billion-parameter fine-tuned
targets, WikiMIA pretrained targets, XSum, WikiText-103, full length sweep,
fixed-prefix sweep, or defenses such as DP-SGD. The reported source does not
fully determine optimizer/LoRA/proposal settings for exact replication. These
are new controlled benchmarks, not evidence about billion-parameter LLMs.

## Run

Install `torch numpy matplotlib tokenizers==0.21.4`. The download flag retrieves
the public sources and verifies SHA-256; alternatively place the two source
files listed in `run_spw_benchmark.py` in `results/spw_scale/data/`.

```sh
python -m unittest discover -s experiments -p test_spw_methods.py
python experiments/run_spw_benchmark.py --download-data --output results/spw_scale
python experiments/report_spw_benchmark.py --input results/spw_scale
```

Data hashes, tokenized split hash, configuration, membership split hash,
checkpoints, per-example scores, trajectory arrays, bootstrap metrics, and
plots are saved under that output directory. Completed cells resume without
retraining; use a fresh output directory after changing configuration/code.
Dataset and generated checkpoint artifacts are not committed to GitHub.
