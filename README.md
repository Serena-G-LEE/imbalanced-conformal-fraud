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
│   └── ieee_cis.yaml            # IEEE-CIS config (transaction + identity join), same schema
├── src
│   ├── __init__.py              # package metadata
│   ├── conformal.py             # StandardSplitCP, MondrianCP, score and quantile helpers
│   ├── data.py                  # chronological 3-way split; ULB and IEEE-CIS loaders
│   ├── metrics.py               # PR-AUC, ROC-AUC, coverage / triage metrics, CSV IO, seed aggregation
│   ├── models.py                # XGBoost / LightGBM behind one train-and-score interface
│   ├── utils.py                 # seeding, YAML config loading, output directories
│   └── visualize.py             # alpha vs. fraud-coverage / review-rate trade-off figures
├── tests
│   ├── test_conformal.py        # quantile, four set types, class-conditional coverage (numpy only)
│   ├── test_metrics.py          # PR-AUC / ROC-AUC vs. brute force, coverage accounting
│   └── test_data.py             # temporal split arithmetic, chronological ordering
├── run_experiment.py            # CLI: models × seeds × methods × alphas -> CSVs + figures
├── requirements.txt
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
`matplotlib`, `xgboost` and `lightgbm` are needed for the full pipeline.

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
| `--models` | all in config | subset of configured model names |
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
    params: {n_estimators: 400, max_depth: 4, learning_rate: 0.05}
experiment:
  seeds: [42, 43, 44, 45, 46]
  alpha_grid: [0.01, 0.03, 0.05, 0.10]
  methods: [standard, mondrian]
  output_dir: results
```

## Evaluation Metrics

| Metric | Definition | Reported in |
| --- | --- | --- |
| `pr_auc` | average precision of the fraud probability on the test block | `per_seed_metrics.csv`, `summary_metrics.csv` |
| `roc_auc` | ROC-AUC (Mann-Whitney, ties = 0.5) | same |
| `coverage` | share of test transactions whose true label is inside the prediction set | same |
| `coverage_normal`, `coverage_fraud` | the same computed inside each class — the quantity Mondrian CP protects | same |
| `triage_rate` | share of transactions in `{0, 1}`, i.e. the manual review workload | same |
| `empty_rate` | share of transactions in `{}` | same |
| `confident_fraud_rate`, `confident_normal_rate` | share of auto-declines and auto-approvals | same |
| `singleton_fraud_precision` / `_recall` | precision and recall of the `{1}` set alone | same |
| `singleton_normal_precision` / `_recall` | precision and recall of the `{0}` set alone | same |
| `fraud_triage_capture` | share of true frauds routed to review rather than auto-declined | same |
| `mean_set_size` | average cardinality of the prediction sets | same |
| `coverage_*_mean/std`, `triage_rate_mean/std` | mean ± standard deviation across seeds | `summary_metrics.csv` |

`triage_rate` is the operational cost curve: at a fixed fraud-coverage target, the calibration scheme
with the lower `triage_rate` is the one an operations team wants.

## Expected Outputs

```
results/<dataset>/
├── per_seed_metrics.csv     # one row per model × seed × method × alpha (31 columns)
├── summary_metrics.csv      # mean / std across seeds for every `(model, method, alpha)`
├── run_metadata.json        # config used, split sizes, prevalence, timestamp, elapsed time
└── figures/
    └── <dataset>_<model>_alpha_tradeoff.png
```

`summary_metrics.csv` is the table to read. For each `(model, method, alpha)` it reports
`coverage_fraud_mean/std`, `coverage_normal_mean/std`, `coverage_mean/std`, `triage_rate_mean/std`,
`empty_rate_mean/std`, `fraud_triage_capture_mean`, `mean_set_size_mean`,
`singleton_fraud_precision_mean`, `singleton_fraud_recall_mean`, `pr_auc_mean/std` and
`roc_auc_mean/std`.

Each figure has two panels: empirical fraud coverage against the nominal `1 − α` reference line on the
left, and the share of transactions in `{0, 1}` on the right, both with seed-standard-deviation error
bars. The expected reading is a flat standard-CP curve far below the diagonal, a Mondrian curve that
tracks `1 − α`, and a review rate that rises as `α` is tightened — the cost of keeping the guarantee.

## Tests

```bash
python3 -m unittest discover -s tests
```

The suite covers the finite-sample quantile (order statistic, ceiling rule, `1.0` clip, empty stratum),
hand-computed thresholds for both calibration schemes, all four prediction-set outcomes, class-conditional
coverage on a 0.3 %-positive synthetic problem (pooled CP collapses on fraud, Mondrian CP does not),
PR-AUC and ROC-AUC against brute-force pairwise computation, the CSV / aggregation layer, and the
temporal split arithmetic. Only `numpy` is required; the pandas-backed loader test is skipped when
pandas is absent.

## Experimental Protocol and Reproducibility

- **Chronological split, not random.** The model trains on the oldest 60 % of transactions, conformal
  quantiles are calibrated on the next 20 %, and every reported number comes from the newest 20 %. A
  random split would leak future information into calibration.
- **Preprocessing is fit on the training block only** (IEEE-CIS frequency encoding, missing-value
  sentinels), so no statistic crosses the split boundary.
- **Seeds affect only the booster** (subsampling, column sampling). Conformal thresholds are
  deterministic given the calibration set, so seed variance reports model variance, not CP variance.
- **Exchangeability.** The finite-sample guarantee assumes exchangeable data; a chronological split
  deliberately breaks it in exchange for an honest evaluation protocol. Empirical coverage is therefore
  the quantity to monitor, and it is reported per class at every `alpha`. The guarantee should be read
  as the design target rather than an unconditional promise under drift — a known tension in the
  time-series CP literature and the natural next extension of this work.

## Limitations and Future Work

- Binary sets only: `{0, 1}` is the whole ambiguity budget. Weighted CP or conformal risk control could
  bound a *cost* (review price × volume) directly instead of a coverage rate.
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
