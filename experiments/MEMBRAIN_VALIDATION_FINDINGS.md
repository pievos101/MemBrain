# Fresh validation findings: output-only Membrain

Verdict: **a promising detector of a high-recovery subset under repeated exposure, not a reliable general training-membership classifier.**

The unchanged16-sample next-word score was frozen before fresh target training.16 target conditions cover two capacities (182,016 and458,592 parameters), two corpora, two exposure regimes and two independent document pools per corpus. 1536 new source sentences include1024 distinct evaluated sentences/4096 candidate-model evaluations. Capacities/regimes share a pool and assignment, so the16 conditions are not independent datasets.

| Score | Fresh macro membership ROC AUC |
|---|---:|
| Membrain 16 | 0.6941 |
| Membrain replicate16 | 0.6814 |
| Membrain pooled32 | 0.6961 |
| Membrain nested3 | 0.6555 |
| Greedy whole-suffix | 0.6182 |
| Length-only control | 0.4775 |
| Min-k20% | 0.9377 |

Min-k20% uses privileged model probabilities; Membrain and the recovery controls use only completion strings. The development0.7106 was exploratory on different data; this fresh study is not a paired numerical comparison to it.

| Capacity / exposure | Primary16 AUC | Greedy whole-suffix AUC |
|---|---:|---:|
| small/standard | 0.5678 | 0.5101 |
| small/repeated | 0.8848 | 0.7490 |
| larger/standard | 0.5450 | 0.5327 |
| larger/repeated | 0.7787 | 0.6810 |

## What is supported

The short-horizon score improves over whole-suffix text recovery, especially for repeated exposure. Primary-versus-greedy paired sample-bootstrap intervals are entirely positive in 9/16 conditions; these are conditional, uncorrected intervals and do not include training/query uncertainty.

| Repeated-exposure unsupervised high group | Mean membership precision | Mean recall |
|---|---:|---:|
| Membrain 16 | 99.3% | 25.5% |
| Membrain replicate16 | 93.0% | 33.3% |
| Membrain pooled32 | 100.0% | 24.9% |

High precision comes with low coverage. It is not a deployment precision/FPR guarantee and is conditional on the benchmark prevalence. The independent query repeat changes some groups substantially. One larger-model repeated condition has high-group Jaccard0.253 despite similar AUC.

## What is not supported

Ordinary training membership remains weak (primary AUC0.568 small/0.545 larger). Larger capacity did not improve performance under the fixed training recipe. Doubling queries to32 did not resolve ordinary membership. Two GMM components appeared even when grouping was near chance; component posterior is not membership probability. Low recovery cannot establish absence from training.

The score is label-free; GMMs fit128 unlabeled discovery examples and evaluate128 different texts, orienting components by recovery mean. All scores and mixtures replay from strings without model weights, a model tokenizer or PyTorch. Only target-model training and retrospective metric calculation use owner-side labels.

Both model capacities remain tiny local models. No external/commercial-LLM claim is established. Repetition changes optimization budget as well as frequency; exact exclusions do not remove semantic overlap. Rare words, an eight-token cap and finite sampling limit the recovery-frequency estimator.

See [frozen protocol](MEMBRAIN_TEXT_VALIDATION.md). Run the validation/reporter to regenerate complete per-target tables, query stability, confidence intervals, grouping, and plots under `results/membrain_text_validation`.66 tests passed.
