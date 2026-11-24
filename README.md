# imbalanced-conformal-fraud

Class-conditional (Mondrian) conformal prediction for triage in extremely imbalanced fraud detection.

Fraud models are usually evaluated by ranking quality (PR-AUC, ROC-AUC), but a deployed system has to
answer a different question for every transaction: approve it, block it, or spend a human reviewer on
it. This repository converts that decision into a calibrated prediction-set problem. A
gradient-boosted fraud model (XGBoost / LightGBM) is wrapped in split conformal prediction, every
transaction receives a set over `{legitimate (0), fraud (1)}`, and the two calibration schemes —
pooled standard split CP versus class-conditional (Mondrian) CP — are compared on their fraud coverage
and on the size of the manual review queue they imply.

The central claim the code is built to test: under extreme imbalance (0.172 % positives in ULB, ≈3.5 %
in IEEE-CIS) a pooled split keeps *marginal* coverage at `1 − α` while its *fraud* coverage collapses,
because the pooled quantile is dominated by the legitimate class; a Mondrian split restores the
class-conditional guarantee `P(Y ∈ C(X) | Y = y) ≥ 1 − α` and thereby makes the review workload a
measured quantity at a chosen coverage level instead of a by-product of threshold tuning.

---

## Highlights

- **Two calibration schemes, one interface.** `StandardSplitCP` and `MondrianCP` share the same
  finite-sample corrected quantile and differ only in how calibration scores are stratified.
- **Triage as a first-class output.** Prediction sets are mapped onto four operational outcomes:
  `{0}` auto-approve, `{1}` auto-decline, `{0,1}` manual review, `{}` reject / fall back.
- **Leakage-free protocol.** Chronological `train / calibration / test` split, preprocessing fit on the
  training block only, model never sees the calibration or test window.
- **Seeded multi-run reporting.** Every `(model, method, alpha)` cell is reported as mean ± standard
  deviation over five seeds, with per-seed CSVs kept for auditing.
- **Two-stage protocol.** A stage-2 calibration ablation compares class re-weighting (`scale_pos_weight`)
  against post-hoc probability calibration (Platt / isotonic) on a leakage-free `prob_cal` split, with
  Brier score and expected calibration error reported next to the triage metrics.
- **Lightweight tests.** The conformal layer is unit tested with `numpy` and `unittest` only, so the
  mathematics can be verified without the boosters installed.

## Overview

### Why a tuned threshold is not enough

The standard recipe — pick the decision threshold that reaches a target recall on a validation set —
has three problems in an imbalanced setting:

1. **No finite-sample guarantee.** The threshold is a point estimate; the recall it produces on the
   next time window is a random variable with no controlled tail.
2. **It hides ambiguity.** When several thousand transactions sit in the 0.4–0.6 score band, a
   threshold must send all of them to one side. There is no way to say "these need a human".
3. **It has to be re-tuned.** Any shift in the score distribution invalidates the previous cut-off, and
   the tuning is done on data that is then also used for reporting.

### What conformal prediction adds

Split conformal prediction replaces the hand-tuned threshold with a quantile of held-out
nonconformity scores and obtains a distribution-free, finite-sample guarantee under exchangeability:

```
P(Y ∈ C(X)) ≥ 1 − α
```

`α` is chosen by the business (e.g. `α = 0.05` means "at most 5 % of frauds may fall outside the
decision rule"), not by the model.

### Why the guarantee has to be class-conditional

The guarantee above is marginal: it averages over both classes. With 0.172 % positives the pooled
calibration quantile is almost entirely determined by the legitimate class, and the fraud
nonconformity scores sit far in the tail of that pooled distribution — they are excluded from the
prediction set even though marginal coverage looks healthy. Class-conditional (Mondrian) calibration
fits one quantile per label stratum,

```
P(Y ∈ C(X) | Y = y) ≥ 1 − α     for every y ∈ {0, 1}
```

at the price of wider sets, i.e. a larger `{0, 1}` review queue. Measuring that price is the point of
this repository.

## Method

**Nonconformity score** (probability of the wrong label):

```
s(x, y) = 1 − p̂(y | x),      p̂(0 | x) = 1 − p̂(1 | x)
```

**Finite-sample corrected quantile** over a stratum of `n` calibration scores:

```
q = s₍ₖ₎,      k = ⌈(n + 1)(1 − α)⌉,      q = 1.0  if k > n  (upper clip)
```

**Prediction set:**

```
C(x) = { y ∈ {0, 1} : s(x, y) ≤ q_y }
```

`q₀ = q₁` for the pooled split, separate `q₀`, `q₁` for the Mondrian split. An empty calibration
stratum (no fraud in the calibration window) falls back to `q_y = 1.0`: the label is never excluded and
every affected transaction is at least reviewed, and the fallback is visible in the output
(`n_cal_fraud = 0`, `min_prob_positive_for_fraud = 0`).

Implementation notes:

- ties are resolved by the observed order statistic (no randomisation), which preserves the coverage
  guarantee and keeps every run deterministic;
- the classifier is fit on the earliest block only, so calibration and test probabilities are strictly
  out of sample;
- `src/metrics.py` implements PR-AUC and ROC-AUC with `numpy` alone, so reported numbers do not depend
  on a scikit-learn version.

## Project Structure

```
imbalanced-conformal-fraud
├── configs
│   ├── ulb.yaml                 # ULB paths, chronological split, seeds, alpha grid, model params
│   ├── ieee_cis.yaml            # IEEE-CIS config (transaction + identity join), same schema
│   ├── ulb_ablation.yaml        # stage-2 sweep: weighted / unweighted x none / platt / isotonic
│   └── ieee_cis_ablation.yaml   # stage-2 sweep on IEEE-CIS, same schema
├── src
│   ├── __init__.py              # package metadata
│   ├── conformal.py             # StandardSplitCP, MondrianCP, score and quantile helpers
│   ├── data.py                  # chronological split (+ optional prob_cal block); ULB and IEEE-CIS loaders
│   ├── metrics.py               # PR-AUC, ROC-AUC, Brier, ECE, coverage / triage metrics, CSV IO
│   ├── models.py                # XGBoost / LightGBM + Platt and isotonic calibrators
│   ├── utils.py                 # seeding, YAML config loading, output directories
│   └── visualize.py             # alpha trade-off figures (per scheme and per ablation variant)
├── tests
│   ├── test_conformal.py        # quantile, four set types, class-conditional coverage (numpy only)
│   ├── test_metrics.py          # PR-AUC / ROC-AUC vs. brute force, coverage accounting
│   ├── test_calibration.py      # Brier, ECE, Platt / isotonic, prob_cal split, leakage guards
│   └── test_data.py             # temporal split arithmetic, chronological ordering
├── run_experiment.py            # CLI: models × seeds × methods × alphas -> CSVs + figures
├── requirements.txt
├── CONTRIBUTING.md              # branching strategy, commit conventions, reproducibility rules
└── README.md
```

Execution flow: `run_experiment.py` reads `configs/<dataset>.yaml`, builds a `SplitDataset` through
`src.data`, trains each configured model per seed with `src.models`, calibrates both CP variants with
`src.conformal`, scores the test block, and writes the tables from `src.metrics` and the figures from
`src.visualize`.

## Getting Started

### 1. Installation

```bash
git clone https://github.com/Serena-G-LEE/imbalanced-conformal-fraud.git
cd imbalanced-conformal-fraud
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.9+ is required. The conformal layer and its tests need only `numpy`; `pandas`, `PyYAML`,
`matplotlib`, `xgboost` and `lightgbm` are needed for the full pipeline. `scikit-learn` is installed as
well because `XGBClassifier` / `LGBMClassifier` refuse to run without it — it is not used by the
conformal or calibration code, which stay numpy-only.

### 2. Prepare Datasets

Data files are not committed (see `.gitignore`); download them into `data/`.

**ULB — Credit Card Fraud Detection (primary).** 284,807 European card transactions over two days, 492
frauds (**0.172 %**). Features are `Time`, 28 anonymised PCA components (`V1..V28`) and `Amount`.

```bash
mkdir -p data/ulb
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ulb --unzip
```

Alternatively download `creditcard.csv` manually from
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>. Expected file:
`data/ulb/creditcard.csv`.

**IEEE-CIS Fraud Detection (extension).** 590,540 transactions (**≈3.5 %** fraud) with transaction and
identity tables, a richer feature set (card, address, device, Vesta aggregates) and a real time axis
(`TransactionDT`).

```bash
mkdir -p data/ieee_cis
kaggle competitions download -c ieee-fraud-detection -p data/ieee_cis --unzip
```

Expected files: `data/ieee_cis/train_transaction.csv` and `data/ieee_cis/train_identity.csv` (downloading
requires accepting the competition rules on Kaggle). The identity table is optional — the loader warns
and continues with transaction features only when it is absent.

| Dataset | Transactions | Fraud rate | Features used | Time column |
| --- | --- | --- | --- | --- |
| ULB | 284,807 | 0.172 % | 30 (`V1..V28`, log1p `Amount`) | `Time` |
| IEEE-CIS | 590,540 | ≈3.5 % | numeric + frequency-encoded categoricals | `TransactionDT` |

### 3. Run Experiments

```bash
# ULB, default grid: 2 models × 5 seeds × 2 methods × 4 alphas
python run_experiment.py --dataset ulb --data-path data/ulb/creditcard.csv

# a smaller sweep
python run_experiment.py --dataset ulb --models xgboost --seeds 42 43 --alphas 0.05 0.10

# IEEE-CIS (transaction + identity join)
python run_experiment.py --dataset ieee_cis --data-path data/ieee_cis

# headless run without figures
python run_experiment.py --dataset ulb --no-figures
```

| Flag | Default | Description |
| --- | --- | --- |
| `--dataset` | `ulb` | `ulb` or `ieee_cis` |
| `--config` | `configs/<dataset>.yaml` | experiment config |
| `--data-path` | from config | ULB: path to `creditcard.csv`; IEEE-CIS: directory holding the CSVs |
| `--output-dir` | `results` | root output directory |
| `--models` | all in config | subset of configured model names **or variant labels** |
| `--seeds` | `42 43 44 45 46` | random seeds |
| `--alphas` | `0.01 0.03 0.05 0.10` | target miscoverage levels |
| `--no-figures` | off | skip PNG generation |

Config keys (both datasets share one schema):

```yaml
data:
  path: data/ulb/creditcard.csv     # or dir_path / transaction_file / identity_file for IEEE-CIS
  time_column: Time
  label_column: Class
  splits: {train: 0.60, calibration: 0.20, test: 0.20}
models:
  - name: xgboost
    scale_pos_weight: auto          # negatives / positives in the training block
    calibrator: none                # none | platt | isotonic
    variant: weighted_raw           # ablation label; defaults to weighted_<calibrator>
    params: {n_estimators: 400, max_depth: 4, learning_rate: 0.05}
  - name: xgboost
    scale_pos_weight: 1.0           # no re-weighting
    calibrator: platt
    variant: unweighted_platt
experiment:
  seeds: [42, 43, 44, 45, 46]
  alpha_grid: [0.01, 0.03, 0.05, 0.10]
  methods: [standard, mondrian]
  output_dir: results
```

`data.prob_cal_fraction` (default `0.15`) is only used when at least one model declares
`calibrator: platt|isotonic`; it is described in the next section.

### 4. Stage-2 calibration ablation

The base run answers "which calibration scheme keeps fraud coverage at the nominal level". It leaves open a
second question: *is the review queue as large as it is because the model's probabilities are on the wrong
scale?* `scale_pos_weight: auto` multiplies the positive class by `negatives / positives` (≈ 578 on ULB),
which minimises a re-weighted likelihood and pushes every predicted probability far above the true fraud
rate. The ablation crosses that choice with post-hoc probability calibration:

```bash
# ULB: 2 boosters x 5 arms x 5 seeds x 2 schemes x 4 alphas
python run_experiment.py --dataset ulb --config configs/ulb_ablation.yaml

# IEEE-CIS (10 arms x 600 trees is long; start with two arms and two seeds)
python run_experiment.py --dataset ieee_cis --config configs/ieee_cis_ablation.yaml \
    --models unweighted_raw unweighted_platt --seeds 42 43 --no-figures

# only the re-weighting question, all seeds
python run_experiment.py --dataset ulb --config configs/ulb_ablation.yaml \
    --models weighted_raw unweighted_raw
```

Arms in both ablation configs (`weighted_raw`, `unweighted_raw`, `unweighted_platt`,
`unweighted_isotonic`, `weighted_platt`):

| Variant | `scale_pos_weight` | `calibrator` | What it isolates |
| --- | --- | --- | --- |
| `weighted_raw` | `auto` | `none` | the stage-1 default: inflated odds, no repair |
| `unweighted_raw` | `1.0` | `none` | the cost of re-weighting, as seen by the score |
| `unweighted_platt` | `1.0` | `platt` | effect of a parametric recalibration on an already-sane score |
| `unweighted_isotonic` | `1.0` | `isotonic` | same, with a flexible (tie-producing) map |
| `weighted_platt` | `auto` | `platt` | repairing the inflated score after the fact |

**Leakage-free `prob_cal` split.** A calibrator is a fitted model, so it may not see the conformal
calibration scores either. When any arm declares a calibrator, the most recent `data.prob_cal_fraction`
(15 %) of the *training* window is carved out as `prob_cal`:

```
train_fit (oldest 85 % of the train block)  ->  booster fit
train_fit .. prob_cal (15 %)               ->  Platt / isotonic fit
calibration (20 %, untouched)              ->  conformal quantiles only
test (newest 20 %)                         ->  every reported number
```

Every arm is fit on the same `train_fit` rows and scored on the same `calibration` / `test` rows, so the
comparison is about re-weighting and calibration, not about how much data each arm saw. The stage-1
configs have no calibrator and keep the original three-block split unchanged.

**How to read it.** Two effects have to be separated, and one of them is structural:

- *Probability quality.* `brier` and `ece` are computed on the test block from the scores the conformal
  layer uses (`brier_raw` / `ece_raw` from the booster output). `weighted_raw` should sit well above
  `*_platt` on both; a Platt fit on the inflated score recovers the true scale, and the isotonic map
  recovers it non-parametrically.
- *Review queue.* `triage_rate` for **Mondrian** CP cannot move under a strictly increasing
  recalibration: each class-conditional threshold is a quantile of one monotone function of `p` per
  stratum, so the threshold moves with the scores and every membership test keeps its answer. This is
  asserted in `tests/test_calibration.py`. Pooled **standard** CP is the exception — its single
  threshold is a quantile of a vector that mixes `p` (legitimate scores) with `1 - p` (fraud scores), so
  a non-linear map reorders it, and its sets (mostly its `empty_rate`) do move. An isotonic fit can also
  move Mondrian sets slightly, because its plateaus create ties that widen the sets.

So the honest reading is: if a `weighted_*` model has a large review queue and its `*_platt` twin has the
same queue with a much lower Brier/ECE, the queue was **not** a probability-scale problem — it is the
model fit (the ranking) that the re-weighting changed, and recalibration alone cannot buy back review
capacity. That is exactly the negative control the stage-2 experiment is for.

## Results: 5-seed Stage-2 ablation at `alpha = 0.05`

The two tables below are the `alpha = 0.05` slice (nominal 95 % coverage) of the stage-2 runs driven by
`configs/ulb_ablation.yaml` and `configs/ieee_cis_ablation.yaml`: ten arms, five seeds (`42–46`) per
cell, every number computed on the newest 20 % test block of a chronological split. Values are the mean
over the five seeds, rounded to three significant figures; the seed standard deviation of every column
sits in the corresponding `*_summary_metrics.csv`.

Raw run artifacts (`results/`) are gitignored, as usual for a benchmark: the CSVs, `run_metadata.json`
and the trade-off figures are regenerated by the commands in this section instead of being committed.
The numbers below are the verified output of those runs.

```bash
python run_experiment.py --dataset ulb --config configs/ulb_ablation.yaml
python run_experiment.py --dataset ieee_cis --config configs/ieee_cis_ablation.yaml
```

Both runs also write the `<dataset>_<model>_ablation_tradeoff.png` figures, which repeat this reading
across the whole `alpha` grid rather than at the single operating point tabulated here.

### ULB (0.172 % fraud, 284,807 transactions)

Mondrian CP, `alpha = 0.05`:

| Model | Variant | PR-AUC | Fraud coverage | Triage rate | ECE |
| --- | --- | --- | --- | --- | --- |
| LightGBM | `weighted_raw` | 0.491 | 96.5 % | 87.3 % | 2.15e-3 |
| LightGBM | `weighted_platt` | 0.447 | 100.0 % | 97.3 % | 5.92e-4 |
| LightGBM | `unweighted_raw` | **0.785** | **98.1 %** | **28.3 %** | 2.03e-4 |
| LightGBM | `unweighted_platt` | 0.785 | 98.1 % | 28.3 % | 2.55e-4 |
| LightGBM | `unweighted_isotonic` | 0.758 | 98.9 % | 65.7 % | 3.35e-4 |
| XGBoost | `weighted_raw` | 0.783 | 92.0 % | 14.9 % | 1.24e-3 |
| XGBoost | `weighted_platt` | 0.783 | 92.0 % | 14.9 % | 1.96e-4 |
| XGBoost | `unweighted_raw` | 0.782 | 96.8 % | 26.2 % | 4.46e-4 |
| XGBoost | `unweighted_platt` | 0.782 | 96.8 % | 26.2 % | 2.36e-4 |
| XGBoost | `unweighted_isotonic` | 0.748 | 100.0 % | 69.2 % | 3.12e-4 |

- **Standard CP collapses on fraud coverage while marginal coverage keeps looking healthy.** Across the
ten arms, marginal coverage at `alpha = 0.05` stays between 95.1 % and 97.5 %, but class-conditional
fraud coverage drops to 0.0 %–59.5 %: five arms (both `*_platt` arms of each booster, plus
`unweighted_raw` on LightGBM) exclude *every* fraud, and only the two `weighted_raw` arms reach
42.7 % / 59.5 %. This is the failure the repository exists to measure — a pooled quantile cannot see a
0.172 % class, so the marginal number is satisfied by the legitimate class alone.
- **On LightGBM, `scale_pos_weight` is what destroyed the ranking.** Disabling it lifts PR-AUC from
0.491 to 0.785 and, at a fixed 95 % target, shrinks the review queue from 87.3 % to 28.3 % (≈3× fewer
reviews) while fraud coverage improves from 96.5 % to 98.1 %. The same swap on XGBoost leaves PR-AUC
unchanged (0.783 → 0.782), so the damage is booster-specific rather than a general law of re-weighting.
- **Recalibration repairs the probability scale, not the queue.** `weighted_platt` on LightGBM improves
ECE 3.6× (2.15e-3 → 5.92e-4) and reaches 100 % fraud coverage, yet the queue *grows* from 87.3 % to
97.3 %: every extra point of coverage is paid for in review capacity, and ECE is not the lever that
sets triage. XGBoost reads the same way on a smaller scale — `weighted_raw` holds a shorter queue
(14.9 %) than `unweighted_raw` (26.2 %) only by missing the nominal target (92.0 % vs 96.8 % fraud
coverage), so the two queue sizes are not comparable at face value.
- **A monotone recalibration leaves Mondrian sets untouched to six decimals.** XGBoost `unweighted_raw`
and `unweighted_platt` share the identical triage rate (`0.261936`) and fraud coverage (`0.968`),
while ECE improves 4.46e-4 → 2.36e-4. This is the invariance asserted in `tests/test_calibration.py`,
now confirmed on real data. Isotonic is the exception: its plateaus create ties, and the triage rate
jumps to 69.2 % (LightGBM 65.7 %, with the widest seed spread in the table) — a flexible map buys
calibration quality by manufacturing ambiguity.

### IEEE-CIS (≈3.5 % fraud, temporal `TransactionDT` split)

Mondrian CP, `alpha = 0.05`:

| Model | Variant | PR-AUC | Fraud coverage | Triage rate | ECE |
| --- | --- | --- | --- | --- | --- |
| LightGBM | `weighted_raw` | 0.471 | 90.6 % | 42.2 % | 7.28e-2 |
| LightGBM | `weighted_platt` | 0.471 | 90.6 % | 42.2 % | 7.11e-3 |
| LightGBM | `unweighted_raw` | **0.485** | 91.1 % | 39.5 % | 7.74e-3 |
| LightGBM | `unweighted_platt` | 0.485 | 91.1 % | 39.5 % | 7.03e-3 |
| LightGBM | `unweighted_isotonic` | 0.468 | 92.2 % | 44.5 % | 7.53e-3 |
| XGBoost | `weighted_raw` | 0.468 | 90.3 % | 43.0 % | 1.20e-1 |
| XGBoost | `weighted_platt` | 0.468 | 90.3 % | 43.0 % | 5.48e-3 |
| XGBoost | `unweighted_raw` | **0.484** | **92.1 %** | **40.6 %** | 5.02e-3 |
| XGBoost | `unweighted_platt` | 0.484 | 92.1 % | 40.6 % | 5.29e-3 |
| XGBoost | `unweighted_isotonic` | 0.468 | 93.1 % | 45.7 % | 5.43e-3 |

- **`unweighted_raw` beats `weighted_raw` on all four columns, for both boosters.** XGBoost: PR-AUC
0.484 vs 0.468, fraud coverage 92.1 % vs 90.3 %, triage rate 40.6 % vs 43.0 %, ECE 5.02e-3 vs
1.20e-1. LightGBM: PR-AUC 0.485 vs 0.471, fraud coverage 91.1 % vs 90.6 %, triage rate 39.5 % vs
42.2 %, ECE 7.74e-3 vs 7.28e-2. There is no trade-off to reason about here — re-weighting costs
ranking quality, coverage and queue length at once.
- **Standard CP again fails the class-conditional target.** Marginal coverage stays at 94.8 %–95.2 %
while fraud coverage is 20.9 % (XGBoost) and 21.2 % (LightGBM) for `unweighted_raw`, and 50.3 % /
59.2 % for the two `weighted_raw` arms.
- **Temporal drift is the visible second effect.** With ≈3.5 % positives the calibration block carries
thousands of frauds, so the quantile is not starved; Mondrian fraud coverage still lands at 91.1 %–
92.1 % for the `unweighted_raw` / `unweighted_platt` arms, 3–4 %p short of the nominal 95 %. Normal
coverage is short on those same arms (94.4 %–94.6 %), which identifies the cause as exchangeability
breaking across `TransactionDT` rather than class imbalance: on ULB, where the file is a two-day draw
with no comparable shift, normal coverage sits *above* nominal for every arm (95.2 %–97.6 %). This is
the empirical motivation for the adaptive /
time-weighted CP in Future Work.
- **Isotonic is not free here either.** It lowers PR-AUC to 0.468 (from 0.484) and widens the queue to
44.5 %–45.7 %; Platt, by contrast, leaves sets identical to `unweighted_raw` while cutting ECE by one
to two orders of magnitude (1.20e-1 → 5.48e-3 on XGBoost, 7.28e-2 → 7.11e-3 on LightGBM).

**Operational reading.** Across both regimes the same ordering holds: use an unweighted booster if the
score feeds a decision rule, keep Mondrian CP for the class-conditional guarantee, and use Platt only
when an honest probability is needed downstream — it will not buy review capacity, and isotonic actively
spends it.

## Evaluation Metrics

| Metric | Definition | Reported in |
| --- | --- | --- |
| `pr_auc` | average precision of the fraud probability on the test block | `<dataset>_per_seed_metrics.csv`, `<dataset>_summary_metrics.csv` |
| `roc_auc` | ROC-AUC (Mann-Whitney, ties = 0.5) | same |
| `brier` | Brier score `mean((p - y)^2)` of the score the conformal layer uses | same |
| `ece` | expected calibration error, equal-mass (quantile) bins, `sum_b n_b/n |acc_b - conf_b|` | same |
| `brier_raw`, `ece_raw` | the same two, computed on the uncalibrated booster output | same |
| `coverage` | share of test transactions whose true label is inside the prediction set | same |
| `coverage_normal`, `coverage_fraud` | the same computed inside each class — the quantity Mondrian CP protects | same |
| `triage_rate` | share of transactions in `{0, 1}`, i.e. the manual review workload | same |
| `empty_rate` | share of transactions in `{}` | same |
| `confident_fraud_rate`, `confident_normal_rate` | share of auto-declines and auto-approvals | same |
| `singleton_fraud_precision` / `_recall` | precision and recall of the `{1}` set alone | same |
| `singleton_normal_precision` / `_recall` | precision and recall of the `{0}` set alone | same |
| `fraud_triage_capture` | share of true frauds routed to review rather than auto-declined | same |
| `mean_set_size` | average cardinality of the prediction sets | same |
| `coverage_*_mean/std`, `triage_rate_mean/std` | mean ± standard deviation across seeds | `<dataset>_summary_metrics.csv` |

`triage_rate` is the operational cost curve: at a fixed fraud-coverage target, the calibration scheme
with the lower `triage_rate` is the one an operations team wants.

`ece` uses equal-mass bins by default (`strategy="quantile"`). At a 0.172 % fraud rate a uniform 15-bin
split over `[0, 1]` puts every transaction in the first bin, where the metric collapses to
`|prevalence - mean score|` and stops seeing the ranking; `src.metrics.ece` still accepts
`strategy="uniform"` for comparison.

## Expected Outputs

```
results/<dataset>/
├── <dataset>_per_seed_metrics.csv   # one row per model × variant × seed × method × alpha (37 columns)
├── <dataset>_summary_metrics.csv    # mean / std across seeds for every `(model, variant, method, alpha)`
├── run_metadata.json                # config, split sizes, prevalence, calibrator fits, timestamp, elapsed
└── figures/
    ├── <dataset>_<model>_alpha_tradeoff.png       # stage 1: scheme comparison
    └── <dataset>_<model>_ablation_tradeoff.png    # stage 2: one line per variant
```

The metric CSVs are prefixed with the dataset name (`ulb_summary_metrics.csv`,
`ieee_cis_summary_metrics.csv`, and the matching `*_per_seed_metrics.csv`), so files copied out of
`results/` stay identifiable. `<dataset>_summary_metrics.csv` is the table to read. For each
`(model, variant, method, alpha)` it reports
`coverage_fraud_mean/std`, `coverage_normal_mean/std`, `coverage_mean/std`, `triage_rate_mean/std`,
`empty_rate_mean/std`, `fraud_triage_capture_mean`, `mean_set_size_mean`,
`singleton_fraud_precision_mean`, `singleton_fraud_recall_mean`, `pr_auc_mean/std`,
`roc_auc_mean/std`, `brier_mean/std`, `ece_mean/std`, `brier_raw_mean/std` and `ece_raw_mean/std`.

Each figure has two panels: empirical fraud coverage against the nominal `1 − α` reference line on the
left, and the share of transactions in `{0, 1}` on the right, both with seed-standard-deviation error
bars. The expected reading is a flat standard-CP curve far below the diagonal, a Mondrian curve that
tracks `1 − α`, and a review rate that rises as `α` is tightened — the cost of keeping the guarantee.
The ablation figure repeats it with one line per variant at a fixed scheme (Mondrian by default), so
`weighted_raw` versus `unweighted_raw` shows what re-weighting does to the queue and `*_platt` overlays
whether recalibration changed it.

## Tests

```bash
python3 -m unittest discover -s tests
```

The suite covers the finite-sample quantile (order statistic, ceiling rule, `1.0` clip, empty stratum),
hand-computed thresholds for both calibration schemes, all four prediction-set outcomes, class-conditional
coverage on a 0.3 %-positive synthetic problem (pooled CP collapses on fraud, Mondrian CP does not),
PR-AUC and ROC-AUC against brute-force pairwise computation, the CSV / aggregation layer, and the
temporal split arithmetic. `tests/test_calibration.py` adds the probability-calibration layer: Brier and
ECE against hand-computed values, monotonicity and `out_of_bounds="clip"` behaviour of both calibrators,
the chronological `prob_cal` split (no overlap with `calibration`, arrays copied), the leakage guards
(`calibrator` without a `prob_cal` block, `X_prob_cal is X_cal`), and the invariance result — a strictly
increasing recalibration leaves Mondrian prediction sets unchanged while the pooled standard threshold
moves. Only `numpy` is required; the pandas-backed loader test is skipped when pandas is absent.

## Experimental Protocol and Reproducibility

- **Chronological split, not random.** The model trains on the oldest 60 % of transactions, conformal
  quantiles are calibrated on the next 20 %, and every reported number comes from the newest 20 %. A
  random split would leak future information into calibration. In the calibration ablation the oldest
  85 % of that training window is the fit block and its last 15 % is `prob_cal`, so the calibrator sees
  only data the booster has already been fit on and that the conformal quantile never uses.
- **Preprocessing is fit on the training block only** (IEEE-CIS frequency encoding, missing-value
  sentinels), so no statistic crosses the split boundary.
- **Seeds affect only the booster** (subsampling, column sampling). Conformal thresholds are
  deterministic given the calibration set, so seed variance reports model variance, not CP variance.
- **Exchangeability.** The finite-sample guarantee assumes exchangeable data; a chronological split
  deliberately breaks it in exchange for an honest evaluation protocol. Empirical coverage is therefore
  the quantity to monitor, and it is reported per class at every `alpha`. The guarantee should be read
  as the design target rather than an unconditional promise under drift — a known tension in the
  time-series CP literature and the natural next extension of this work.

## Branching Strategy

`main` keeps only verified states. Features and experiments are developed on short-lived branches
(`feat/<scope>`, `exp/<dataset>-<study>`) and merged with `--no-ff`, so each change is revertible as a
unit (`git revert -m 1 <merge-commit>`). Dataset-specific experiments branch from `main` as
`exp/ulb-baseline` and `exp/ieee-cis`, carrying only config overrides and result summaries — data and raw
outputs stay out of version control. Full rules, including commit conventions, are in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Limitations and Future Work

- Binary sets only: `{0, 1}` is the whole ambiguity budget. Weighted CP or conformal risk control could
  bound a *cost* (review price × volume) directly instead of a coverage rate.
- Recalibration is only as good as its block: `prob_cal` is 15 % of the training window, which on ULB is
  a few hundred positives and on IEEE-CIS a few thousand. A quantile-of-score diagnostic (vary
  `prob_cal_fraction` and watch the Brier / ECE spread) is the natural follow-up, as is temperature
  scaling fit by cross-validation inside `train_fit`.
- Static calibration: quantiles are fit once on a 20 % window. Online / adaptive CP is the natural
  response to concept drift.
- No label delay: real chargebacks arrive weeks after the transaction, so calibration labels here are
  optimistic.
- Planned: cost-sensitive queue construction (rank `{0, 1}` by score under a daily review budget), a
  per-time-slice coverage diagnostic, and a comparison against cost-tuned threshold baselines at matched
  review volume.

## Citation

If you use this code, please cite the repository:

```bibtex
@misc{lee2026imbalancedconformalfraud,
  title  = {imbalanced-conformal-fraud: Class-conditional conformal prediction for triage in
            extremely imbalanced fraud detection},
  author = {Lee, Serena},
  year   = {2026},
  url    = {https://github.com/Serena-G-LEE/imbalanced-conformal-fraud}
}
```

## References

- Vovk, Gammerman & Shafer. *Algorithmic Learning in a Random World.* Springer, 2005. (Mondrian
  conformal predictors.)
- Angelopoulos & Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free
  Uncertainty Quantification.* arXiv:2107.07511, 2021.
- Lei, G'Sell, Rinaldo, Tibshirani & Wasserman. *Distribution-Free Predictive Inference for Regression.*
  JASA, 2018.
- Dal Pozzolo, Caelen, Johnson & Bontempi. *Calibrating Probability with Undersampling for Unbalanced
  Classification.* IEEE SSCI, 2015. (ULB dataset.)
- IEEE-CIS Fraud Detection, Kaggle, 2019.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
