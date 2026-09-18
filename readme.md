# Kaggle S6E9 — Predicting Electric Vehicle Purchases

**English** | [简体中文](README.zh-CN.md)

Predict the probability of `Will_Buy_EV = Yes`, evaluated with ROC AUC.
This project compares LightGBM, XGBoost and CatBoost through a modular
cross-validation pipeline. Features include digits, multiscale bins, frequency
encoding and target encoding. Out-of-fold (OOF) predictions support comparisons
between individual models and blends.

Full local results are recorded in `artifacts/experiments.csv`. Public leaderboard scores
require an actual submission. This project does not submit to Kaggle automatically.

## Installation and data

Python 3.12 is recommended. Run commands from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Input files:

```text
data/train.csv              # id + 13 original features + Will_Buy_EV
data/test.csv               # id + 13 original features
data/sample_submission.csv  # Kaggle submission format
```

The training set contains 668,665 rows; the test set contains 286,571 rows.
Labels are `No` / `Yes`. The `id` column is used for alignment and excluded from
model inputs. The loader checks unique IDs, valid labels and matching feature columns.

## Architecture

```text
training/
  lightlgm_cv.py           # Historical module spelling remains supported
  lightgbm_cv.py           # Equivalent entry point with standard spelling
  lightgbm_reference_cv.py # Preserved original 10-fold script from 175fba1
  xgboost_cv.py
  catboost_cv.py
  experiment_suite.py     # Sequential candidates followed by blend selection
  ensemble.py             # Blend existing OOF bundles
utils/
  data_loader.py          # EVDataLoader retains its five-value return contract
  feature_engineering.py
  cv_runner.py            # Shared folds, encoding, checkpoints and evaluation
  prediction_saver.py     # Versioned predictions and experiment records
  ensemble.py             # ID alignment, weight selection and diagnostics
predictions/              # Submission CSVs only
artifacts/predictions/    # OOF NPZs, configuration JSONs and blend reports
artifacts/                # Fold checkpoints, logs and progress; excluded from Git
tests/
```

Separate model entry points, CLI overrides, early stopping, OOF AUC, per-fold
scores, best-iteration statistics, prediction versions and experiment records
are retained. Shared training mechanics live in `utils/cv_runner.py` to keep
encoding and validation consistent across models.

## Running experiments

### Existing entry points

```bash
python -m training.catboost_cv
python -m training.xgboost_cv
python -m training.lightlgm_cv
```

CatBoost and XGBoost retain their original raw-feature defaults and principal
parameters. Add `--feature-engineering` to use the `legacy` business features.
LightGBM defaults to the `notebook` recipe, 10 folds and triple target encoding.

```bash
python -m training.catboost_cv --depth 4 --learning-rate 0.05 --iterations 5000
python -m training.xgboost_cv --feature-engineering
python -m training.lightgbm_cv --no-target-encoding
```

### Preserved original LightGBM

```bash
python -m training.lightgbm_reference_cv
```

This entry point preserves the training configuration from `175fba1`: 10 folds, learning rate
0.005, up to 100000 iterations, early-stopping patience 500, digit/frequency
features and triple target encoding. It still calls shared data, feature and
prediction utilities; the corresponding feature formulas are unchanged.
The `strong` preset does not replace this script.

Like the earlier version, it saves a submission and experiment record only after
all folds finish. It does not provide the new fold checkpoints or OOF bundles.
Previously generated predictions remain available.

### Additional candidates

```bash
python -m training.lightgbm_cv --preset strong
python -m training.xgboost_cv --preset strong
python -m training.catboost_cv --preset strong
```

`strong` is a candidate configuration, not a claim of optimality. It enables
multiscale features, fold-local frequency and triple target encoding,
income-neighborhood encoding, and five folds. CPU and four threads are the defaults.

| Parameter | LightGBM strong | XGBoost strong | CatBoost strong |
|---|---:|---:|---:|
| Maximum iterations | 3500 | 2400 | 3500 |
| Learning rate | 0.02 | 0.03 | 0.05 |
| Depth | 5 | 6 | 6 |
| Early-stopping patience | 150 | 150 | 200 |
| Feature subsampling | 0.30 | 0.55 | 0.80 |

Explicit CLI arguments override presets. `--model-seed` changes model randomness;
`--random-seed` changes fold and encoding randomness.

### Full comparison and blending

```bash
python -m training.experiment_suite --n-splits 5 --n-jobs 4
```

The suite runs sequentially:

1. Multiscale LightGBM.
2. Digit/frequency LightGBM with learning rate 0.02, up to 20000 iterations and early-stopping patience 300.
3. Multiscale XGBoost.
4. Multiscale CatBoost.
5. OOF-based selection among single models, probability blends and rank blends.

All four use the same rows, folds and split seed. The suite's `notebook` candidate
uses learning rate 0.02, unlike the preserved 0.005 configuration. A full run can
take several hours.

```bash
# Run only two candidates
python -m training.experiment_suite --candidates multiscale_lgb multiscale_xgb

# Small smoke run; not a full competition evaluation
python -m training.experiment_suite --sample-size 6000 --n-splits 2 --max-iterations 20
```

## Feature recipes

| Recipe | Features | Where fitted |
|---|---|---|
| `legacy` | Charging, commute, income and categorical business interactions | Row-wise |
| `notebook` | Income/charging interactions, digits -4…3, one-hot encoding, frequencies, constant/perfect-correlation filtering | Label-free preprocessing; frequencies use train + test |
| `multiscale` | Income digits/remainders and multiple income/commute bin widths; original categories retained | Row-wise; frequencies fitted on each outer training fold |

The `notebook` recipe preserves floating-point floor division for digit features;
rounding before extraction changes their values. The multiscale recipe explicitly
rounds commute distance multiplied by 10, providing a different representation.

**All label-dependent features are fitted inside outer training folds:**

- Triple target encoding uses smoothing `auto`, `10` and `100`. Training rows use
  five-fold inner cross-fitting through `fit_transform`; validation and test rows
  only use `transform`. The `notebook` default encodes seven numeric columns;
  multiscale encoding uses original columns and selected bin keys.
- `--te-scope all` optionally includes digit and original categorical columns for
  non-multiscale recipes.
- `--income-neighbors` uses income resolutions 8192 and 16384 to estimate central,
  neighboring, left and right purchase rates, slopes, curvature and counts.
  Training rows are cross-fitted; missing values fall back to the training prior.

The `notebook` recipe uses the unlabeled test distribution for frequencies. It is
transductive competition preprocessing, not strictly train-only preprocessing.
Constant/correlation filtering also precedes outer CV. Multiscale frequencies,
category vocabularies and label statistics are fitted on outer training rows.

```bash
# Controlled feature ablations
python -m training.lightgbm_cv --preset strong --no-income-neighbors
python -m training.lightgbm_cv --preset strong --no-target-encoding --no-income-neighbors
python -m training.lightgbm_cv --feature-recipe legacy --no-target-encoding
```

Keep folds consistent when comparing models or features. Hyperparameter and
candidate selection introduce selection bias; small gains do not establish
stable generalization improvements.

## Outputs, progress and recovery

Each completed fold produces a checkpoint:

```text
artifacts/runs/<model>_<signature>/
  config.json
  progress.json
  fold_01.npz
  ...
  result.json           # Created only after all folds finish
```

The signature includes input content hashes, configuration, core source-file
hashes and dependency versions. Repeating the same command reuses completed folds
by default; an interrupted fold restarts. Use `--no-resume` to retrain.
Source changes, including comments, can change the signature, as can parameter or
dependency changes. Incompatible checkpoints are not reused.
Checkpoints contain predictions, not deployable models. Repeating a completed run
may create another submission version.

After full CV, files share a prefix but are stored separately:


Generated files are routed by the code: submission CSVs go to `predictions/`, metadata and OOF bundles to `artifacts/predictions/`, experiment records to `artifacts/experiments.csv`, and checkpoints/logs to `artifacts/runs/` or `artifacts/suite/`. CatBoost file logging is disabled. Already-running processes keep their old output settings; these defaults apply on the next launch.

- `predictions/*.csv`: `id,Will_Buy_EV`, ready for Kaggle submission.
- `artifacts/predictions/*.npz`: training IDs, labels, OOF predictions, fold IDs, test IDs and predictions.
- `artifacts/predictions/*.json`: configuration, scores and blend reports. `artifacts/experiments.csv` also receives a record,
  accommodating different model fields.

Incomplete bundles, duplicate IDs, inconsistent labels/folds and invalid
probabilities are rejected. Historical submission-only runs cannot recover OOF
predictions without retraining. Sample runs cannot be blended with full-data OOF.

The suite also writes `artifacts/suite/suite_status.json` and per-candidate logs.
Modern model entry points and the suite support `--train-path`, `--test-path`,
`--output-dir`, `--artifact-dir`, `--experiment-file` and `--run-root`. The preserved original entry
point does not support fold recovery. Use `--help` for each command's options.

## Ensemble evaluation limits

```bash
python -m training.ensemble artifacts/predictions/model_a.npz artifacts/predictions/model_b.npz
```

Inputs are aligned by ID and must share training rows, labels and folds.
Candidates include single models, equal-weight blends and pairwise weights of
25%, 50% and 75%, using probabilities or ranks. Weights are selected on one fixed
half of the OOF rows; the other half is reported without further weight tuning.
A single model may win.

**This is an OOF diagnostic, not independent nested validation.** Base learners
trained on other folds may have seen labels from the other half. The full OOF
score is also affected by blend selection, so reports label it
`full_oof_auc_after_selection`. Strict ensemble evaluation requires a completely
isolated holdout or fully nested retraining.

Rank blends target AUC and do not produce calibrated purchase probabilities.
The pipeline does not incorporate test-only external predictions without
verifiable OOF, perform leaderboard-driven weight tuning, or use pseudo-labels.

## Tests and results

```bash
python -m unittest discover -s tests -v
```

Tests cover target-encoding cross-fitting, unseen-value fallback, neighborhood
encoding, ID/fold alignment, parameter overrides and output validation.
Small end-to-end checks cover all four candidates, checkpoint recovery, prediction
saving and blending.

Historical full-data baselines, whose settings may differ from newer experiments:

| Experiment | OOF AUC |
|---|---:|
| CatBoost depth 4, raw features | 0.9419543 |
| CatBoost depth 4, business features | 0.9417906 |
| XGBoost, business features | 0.9416650 |
| LightGBM, business features | 0.9417209 |

New full-data results are recorded in `artifacts/experiments.csv` and suite `result.json` files.

## Local validation and handoff — 2026-09-17

The integrated pipeline passed 11 tests and small four-candidate training,
blending and recovery checks. Full training was paused at the user's request
for local continuation:

```bash
source .venv/bin/activate
python -m training.experiment_suite --n-splits 5 --n-jobs 4
```

This uses the same training configuration as the started suite. Completed folds
are reused only when their signatures also match; incomplete folds restart.
Keep `artifacts/suite/`. Later source or dependency changes can invalidate reuse.

| Completed full-data experiment | Outer folds | OOF AUC | Submission |
|---|---:|---:|---|
| Preserved digit/frequency LightGBM | 10 | 0.94580 | v017 |
| Multiscale + neighborhood + triple-TE LightGBM | 5 | 0.94612535 | v018 |

Fold counts differ, so this comparison alone does not establish a stable gain.
The matched five-fold `notebook`, XGBoost, CatBoost and final blend have not all
completed; no full-data ensemble improvement is claimed.

Both completed submissions remain in local `predictions/`. Deleted OOF bundles
can be regenerated from completed fold checkpoints when configuration, code and
dependency signatures match. The original script is retained separately in
`training/lightgbm_reference_cv.py`.

Confirmed public leaderboard score: **0.94639** for v018, based on the Kaggle
submission result supplied by the user on 2026-09-17.

## Acknowledgements

The `notebook` feature recipe and the preserved reference LightGBM
configuration were adapted from Rugved Bane's Kaggle notebook:

"0.94590 LB — Stacking failed, this didn't"
[https://www.kaggle.com/code/rugvedbane/0-94590-lb-stacking-failed-this-didn-t](https://www.kaggle.com/code/rugvedbane/0-94590-lb-stacking-failed-this-didn-t)

In particular, the reference approach motivated:

- digit decomposition
- frequency encoding
- triple target encoding
- the reference LightGBM configuration

This repository extends that baseline with a reusable CV pipeline,
leakage-safe fold-local feature fitting, OOF artifacts, checkpoint/resume,
multi-scale features, income-neighborhood encoding, model comparisons,
ensemble diagnostics, and automated tests.

## Next experiments from v18

The v20 run reproduces v18. Its missing OOF bundle can be recovered without training:

```bash
python -m training.recover_oof artifacts/suite/multiscale_lgb/lightgbm_0f835db60a41eb8c
```

Recovery verifies input data hashes, folds and predictions against the saved submission. It supports completed full-data runs only.

```bash
# No model training: compare v20 (= v18) with small v22 weights.
python -m training.ensemble artifacts/predictions/prediction_v020_*.npz artifacts/predictions/prediction_v022_*.npz --challenger-weights 0.1 0.2 0.3

# Optional full training: same folds and features, different model seeds.
python -m training.lightgbm_cv --preset strong --n-splits 5 --random-seed 42 --model-seed 17 --n-jobs 4
python -m training.lightgbm_cv --preset strong --n-splits 5 --random-seed 42 --model-seed 2026 --n-jobs 4
```

Inspect `audit_gain_over_baseline` in the blend report before submitting. This is an OOF diagnostic, not independent nested validation; gains are not guaranteed. New seed runs produce new versioned bundles; use their exact paths with `training.ensemble` to compare them with the baseline.
