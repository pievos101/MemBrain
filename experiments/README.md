# Membrain experiments

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
