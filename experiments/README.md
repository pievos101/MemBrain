# Membrain experiments

## Simplified Membrain experiment: two embedding curves plus absolute loss

At each increasing true prefix, retain one embedding of the true suffix and
one embedding of the generated suffix. Current Membrain uses the fixed score
`-A[teacher CE + embedding distance * absolute free-running true-token CE]`.
No attack classifier, membership labels, fitted weights, Min-k anchor, SPW token
weighting, or relative rollout-loss gap is used in this simplified primary score.
The full-dimensional embedding vectors and continuation curves are saved. This
is a research candidate; a simpler formula is not assumed to improve detection.

```bash
python -m unittest discover -s experiments -p test_membrain_simple.py
python experiments/run_membrain_simple.py
python experiments/report_membrain_simple.py
python experiments/score_membrain_simple.py \
  --target-checkpoint results/membrain_simple/cells/ag_news_6101_standard/target.pt \
  --text-file /path/to/sample.txt
```

See [MEMBRAIN_SIMPLE_PROTOCOL.md](MEMBRAIN_SIMPLE_PROTOCOL.md) for the fixed
definition, fresh-data exclusions, target training, and indispensable loss-only
controls. The origin classes true=0/generated=1 do not become membership labels.
No score is a calibrated membership probability. Earlier public artifacts and
chunk archives are needed for target pretraining and exact data exclusion.

## Earlier fixed-score Membrain with a Min-k anchor

Membrain now uses a fixed per-text score. **No membership labels, shadow
classifier, origin classifier, learned normalizer, or probability calibration
are used by the attack.** It retains whole generated continuations at increasing
true prefixes, adds SPW-inspired position weights and a within-text alignment
control, and combines recovery support with Min-k20% using a fixed formula.

```bash
python -m unittest discover -s experiments -p test_membrain_unsupervised.py
python experiments/run_membrain_unsupervised.py --download-data
python experiments/report_membrain_unsupervised.py
```

See [MEMBRAIN_UNSUPERVISED_PROTOCOL.md](MEMBRAIN_UNSUPERVISED_PROTOCOL.md) for
the exact score and fresh twelve-target benchmark. The benchmark needs the
public tokenizer/background/proposal artifacts from the first SPW comparison
and raw chunk archives from the historical experiments to exclude used texts.
Membership labels are used to train the target LMs and evaluate attacks, never
as inputs to or tuning data for the Membrain score. Outputs are ranking scores,
not membership probabilities. No binary threshold is fitted.

```bash
python experiments/score_membrain_unsupervised.py \
  --target-checkpoint results/membrain_unsupervised/cells/ag_news_5101_standard/target.pt \
  --text-file /path/to/sample.txt
```

## Historical supervised revision (not current Membrain)

The archived supervised experiment retains whole generated suffixes at increasing true prefixes, adds
aligned and SPW-weighted embedding comparisons, and learns **membership** on
separate shadow models rather than only learning trajectory origin. Development,
validation, probability calibration, and fresh target documents are disjoint.

```bash
python -m unittest discover -s experiments -p test_membrain_prefix.py
python experiments/run_membrain_prefix.py --stage development
python experiments/run_membrain_prefix.py --stage test
python experiments/report_membrain_prefix.py
```

See [MEMBRAIN_PREFIX_PROTOCOL.md](MEMBRAIN_PREFIX_PROTOCOL.md) for the fixed
protocol, baseline reruns, equal-supervision ablations, and probability/access
assumptions. It runs 20 shadow models and 12 fresh benchmark targets. Prepare
the previous suite's source/tokenizer/background/proposal artifacts first using
the workflow below. All models remain small CPU Transformers.

Score a new text using one of the compatible local target checkpoints:

```bash
python experiments/score_membrain_prefix.py \
  --target-checkpoint results/membrain_prefix/cells/target_ag_news_4101_standard/target.pt \
  --text-file /path/to/sample.txt
```

The scorer uses the first 48 BPE tokens, reports all prefix curves, and returns
a membership score calibrated under the balanced shadow prior. Only load trusted
local attack checkpoints. Calibration does not establish probabilities for
arbitrary external models, domains, or membership priors.

## SPW-MIA comparison on new natural-text benchmarks

The latest comparison independently implements SPW-MIA and the five baselines
in Xie et al. (KBS 2026), and adds Membrain. It trains eight small causal
Transformers on AG News and WikiText-2 with ordinary/repeated exposure. These
are **new CPU-scale benchmarks**, not the paper's billion-parameter numerical
reproduction. Neighbor uses a documented local masked-model proposal adaptation.

```bash
uv pip install --python .venv/bin/python tokenizers==0.21.4
python -m unittest discover -s experiments -p test_spw_methods.py
python experiments/run_spw_benchmark.py --download-data
python experiments/report_spw_benchmark.py
```

See [SPW_BENCHMARK.md](SPW_BENCHMARK.md) for exact data splits, algorithms,
access assumptions, inference score definition, and evaluation limitations.
Outputs include per-example scores, model checkpoints, bootstrap intervals,
ROC/PR and prefix-recovery plots, and a Markdown report under `results/spw_scale`.
The primary Membrain membership score is the negative absolute difference
between true/generated trajectory-origin probabilities. It is not a calibrated
membership probability; the origin CNN never trains on membership labels.

## Earlier trajectory workflow

The latest workflow tests whether repeated exposure lets a language model recover
a sentence from shorter prefixes, then learns an embedding-trajectory classifier:

- Class 0: true prefix followed by the true continuation.
- Class 1: the same true prefix followed by a generated suffix.

Classifier probabilities describe these origin classes. They are not calibrated
training-membership probabilities. Membership/exposure labels are used only for
evaluation in the latest trajectory-classifier experiment.

## Environment and data

Run from the repository root. Tested with Python 3.12, NumPy 2.2.6,
PyTorch 2.6.0+cpu and Matplotlib 3.10.8. For example, using `uv`:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy==2.2.6 matplotlib==3.10.8
source .venv/bin/activate

mkdir -p results/membrain_loss_curves/source
curl --fail --location https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
  --output results/membrain_loss_curves/source/tinyshakespeare.txt
echo '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed  results/membrain_loss_curves/source/tinyshakespeare.txt' | sha256sum --check
```

The retained source checksum is also verified by the data-loading code. No API
keys, external services or GPU are required. Set writable cache directories if
your environment has a read-only home directory.

## Latest workflow

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/membrain-matplotlib

python experiments/membrain_recovery.py
python experiments/report_membrain_recovery.py
python experiments/membrain_trajectory_classifier.py
python experiments/report_membrain_trajectory_classifier.py
python -m unittest discover -s experiments -p 'test_*.py'
```

Run recovery before the trajectory classifier and full test suite: they use its
retained sentence splits and model checkpoints. Recovery trains four character
GRUs, with a balanced exposure rotation over 128 Shakespeare sentences. Actual
training counts are 0, 40, 160 and 640 presentations. Complete remaining sentences
are generated from increasing prefixes; a fixed final-segment control is included.

The trajectory classifier uses additional disjoint auxiliary sentences, separate
training/stopping/calibration/holdout sets, and full-suffix hidden trajectories.
Pooled and single-prefix controls and the LOSS comparison are included.

Score a new sentence with the saved LM/classifier pair:

```bash
python experiments/membrain_trajectory_classifier.py \
  --text-file results/membrain_trajectory_classifier/example_new_sentence.txt \
  --model-seed 51001 --variant trajectory
```

This experimental scorer expects one 80–120 character printable-ASCII sentence,
matching its training domain. It returns both origin-class probabilities and a
paired reference score.

## Reviewer-directed test: information beyond LOSS

After preparing the recovery and trajectory-classifier manifests above:

```bash
python experiments/membrain_beyond_loss.py
python experiments/report_membrain_beyond_loss.py
```

This uses fresh, disjoint role-specific sentence pools and ten independent LMs
at three exposure checkpoints. Origin networks remain membership-label-free.
Separate supervised shadow probes test incremental information over flexible
LOSS on held-out models, with flattened/shuffled/pooled and loss/recovery controls.
Loss matching uses only loss and labels, never Membrain scores. Raw LOSS is also
retained with its conventional ranking. The original analysis plan, execution
configuration and locally recorded amendments distinguish primary and exploratory
analyses. `--analysis-only` reuses retained models/scores. Checkpoint and feature
caches can be large; allow several minutes and use a fresh output after changing
model/feature definitions. The primary five-epoch test did not establish a gain
over LOSS in this run; the code does not assume an improvement.

## Earlier comparisons

- `corrected_benchmark.py`: disjoint synthetic benchmark and embedding affinity.
- `membrain_geometry.py`: region diagnostics and shadow membership calibration.
- `membrain_loss_curves.py`: teacher-forced versus generated self-likelihood curves.
- `membrain_rollout.py`: free-running cross-entropy against **true** continuation
  tokens, compared with teacher-forced LOSS. This differs from self-likelihood.
- Corresponding `report_*.py` scripts generate figures and numerical reports.

Results, downloaded data, checkpoints, plots and the LaTeX paper are not included
in this code-only publication. The commands regenerate outputs under `results/`.
Matching caches may be reused; clear the affected generated output directory or
choose a fresh output when changing model/feature code. Downstream scripts use
the default result directories unless their source explicitly supports another.
The original `test_final_v1.py` through `test_final_v6.py` remain unchanged.
