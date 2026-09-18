# Original-data feature experiment

## Source and permission check

The [competition data page](https://www.kaggle.com/competitions/playground-series-s6e9/data) links directly to [EV Adoption Behavior and Range Anxiety, by itzzomkar](https://www.kaggle.com/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety). The source itself is synthetic, not a verified collection of real buyers.

Checked on 2026-09-17:

- Competition-specific [rule 2.6](https://www.kaggle.com/competitions/playground-series-s6e9/rules) permits public external data that is equally accessible, subject to the stated conditions. This was verified on the rendered competition rules page, not inferred from another competition.
- Kaggle's dataset metadata API reports version 1, updated 2026-07-06, and `CC0: Public Domain`. This source license is distinct from the competition dataset's CC BY 4.0 license.
- The public download needs no paid access. Source data is not committed to this repository; the download helper verifies its SHA-256.
- CSV SHA-256: `c271a380df51b18177d0a039d54525ca7c3d71500701dc7496714a091df16fae`.

## Data audit

The source has 10,000 rows and a 17.5% positive rate, versus 17.4645% in competition training. All 13 competition features are available; `Buyer_ID` is excluded. Missing source values: income 178 rows, commute 181, environmental concern 184. No exact source feature duplicates or exact source-to-competition train/test matches were found after numeric dtype normalization. This does not rule out near duplicates or dependence introduced by the synthetic generator.

Source exact-value coverage of competition training is 97.9449% for income, 99.9996% for commute, and 100% for the other features. Income coverage on competition test is 97.9185%. Coverage motivates testing external statistics; it does not establish predictive improvement.

Machine-readable audit and metadata: `artifacts/external_data/`. Source CSV: `data/external/ev_adoption.csv`.

## Controlled implementation

Enable `--original-data` to add exactly 13 columns, `original_TE_<feature>`. Each maps a feature value to a smoothed target mean learned only from retained source rows:

`(source_positive_count + 20 * source_prior) / (source_count + 20)`

Smoothing 20 is fixed in advance, not tuned on the leaderboard. Missing or unseen keys receive the retained source prior. All source rows whose complete feature vector matches any competition train/test row are removed before fitting. All duplicate source feature groups are also removed. Competition targets are never used by this encoder, and competition rows are never appended to the external labeled data.

The exclusion examines the full supplied training CSV even during sampled development runs. It uses feature values only. Existing competition target encoders keep their inner cross-fitting; external columns are appended afterward so the existing feature recipe and its encoder keys are preserved. No new data rows, folds, hyperparameters, seed, or model family are introduced. Without the flag, the modeling behavior remains the baseline.

The source file hash and external encoder code hash participate in the run identity, and an audit is saved in each run directory. Source-file or code changes invalidate cached runs. Existing historical artifacts remain available; do not rerun v18 merely to obtain a matching new code signature.

## Run

The source has already been prepared on the current machine. Train one full five-fold experiment:

```bash
python -m training.lightgbm_cv --preset strong --original-data --n-jobs 4
```

On a fresh checkout, first download and audit the pinned source (no training):

```bash
python -m training.prepare_original_data
```

An optional explicit source path can follow `--original-data`, but it must match the pinned, audited CSV checksum. Current integration requires the multiscale feature recipe. Predictions retain the existing directory layout: only submission CSVs in `predictions/`; JSON, OOF, checkpoints and audit reports in `artifacts/`. Submission names include `original`.

## Validation and decision

Unit tests cover exact-overlap exclusion, duplicate/conflicting source removal, independence from competition labels, smoothing, unknown/missing fallback, dtype normalization, checksum validation and the opt-in defaults. A two-fold, three-tree, 400-row development smoke run passed; its artifacts are isolated under `artifacts/smoke/original/`. No full-data training or public submission has been run for this change.

Compare the new single-model OOF and paired fold scores with v18/v20 first (same model seed 42 and five-fold split), then with v27 as the current ensemble benchmark. Review all folds rather than cherry-picking Fold 3. No score improvement is claimed. If the result is worse or offers no independently supported incremental value, retain v27 and stop this direction instead of sweeping source smoothing or repeatedly tuning blend weights on the same audit rows.
