# Public approach review — 2026-09-17

Only publicly downloadable notebooks were inspected. This is a review of a
selection of relevant public approaches, **not all leaderboard teams' code**.
Leaderboard positions and notebook titles change; reported scores below are
source claims, not scores achieved by this repository. Downloaded notebooks were
read as data; their installation cells and external code were not executed.

## Incorporated techniques

| Source | Inspected version | What informed this project | Local adaptation |
|---|---:|---|---|
| [Rugved Bane — Stacking Failed, This Didn't](https://www.kaggle.com/code/rugvedbane/0-94590-lb-stacking-failed-this-didn-t?scriptVersionId=349757944) | exact script 349757944 | FE-B, digit decomposition, train+test frequencies, triple TE, LightGBM | Modular implementation, CPU default, shared dummy vocabulary, new sklearn/LightGBM APIs |
| [jazivxt — Single Model Zoom Zoom](https://www.kaggle.com/code/jazivxt/single-model-zoom-zoom) | 9 | Income/commute multiscale keys, outer-fold frequencies, cross-fitted income-neighborhood statistics, model diversity | Standard stratified folds, no external prediction archives or reference-based restratification, no hard-coded target-derived income thresholds; independently implemented neighborhood encoder |
| [Kirill — S6E9 LightGBM](https://www.kaggle.com/code/kirill0212/s6e9-lightgbm) | 2 | Narrow feature subsampling, shallow trees, explicit TE of raw feature keys | Uses existing sklearn TargetEncoder rather than adding catstat/neural dependencies; shared encoding infrastructure |
| [Mikhail Naumov — Electric Vehicle Purchases / XGB](https://www.kaggle.com/code/mikhailnaumov/electric-vehicle-purchases-single-xgb) | see inspection record below | Explicit target encoding of digit/category keys, numeric bin representations | Optional `--te-scope all`; no external original-data target aggregates, no pretrained output dependency |
| [Vinay Rai — Rank Averaging Ensemble](https://www.kaggle.com/code/vinay24baghira/s6e9-rank-averaging-ensemble-0-946-lb) | 1 | Rank averaging for an AUC competition | Require aligned local OOF + test predictions, validate IDs and folds; never silently skip missing inputs |
| [miickey — Multi-Scale Rank Transfer Blend](https://www.kaggle.com/code/miickey/0-94637-lb-multi-scale-rank-transfer-blend) | 9 | Separate blend selection and audit, compare individual sources and correlation | Small fixed weight grid, OOF diagnostic explicitly distinguished from nested CV; no transfer of weights to unrelated test-only predictions |

The first reference reports OOF 0.94580 / LB 0.94590. The rank-averaging page
reports public 0.94624. These values are included only to identify the references.
The examined miickey code records a 0.94640 transfer candidate, which does not
constitute independent confirmation of its score or a result of this repository.

## Reviewed but not copied into a training pipeline

- [Aman Atar — Smart Weighted Rank Ensemble](https://www.kaggle.com/code/amanatar/s6e9-smart-weighted-rank-ensemble), v13: combines external submission files with hand-set weights. It does not train those component models. A monotonic power transform of one final score does not improve ROC AUC by itself.
- [najiama — Electric Vehicle OOF CV 0.94618 LB 0.94633](https://www.kaggle.com/code/najiama/s6e9-electric-vehicle-oof-cv-0-94618-lb-0-94633), v1: loads a prepared OOF/submission dataset. The notebook alone does not reconstruct its training procedure.
- [chovyxu — EV Adoption Tokens / XGBoost / TinyTokenTransformer](https://www.kaggle.com/code/chovyxu/ev-adoption-tokens-xgboost-tinytokentransformer): inspected feature/preprocessing and model structure. A separate deep-learning stack and GPU-oriented training would be a different experiment; it was not blindly added to the local tree ensemble.

Original-data augmentation, pseudo-labeling, focal-loss custom objectives and
reference-prediction-based fold construction are not enabled. They require their
own data/provenance and fair validation rather than assuming every high-scoring
notebook component improves the same local model.

## Reproducibility and attribution

The architecture and experiment tracking originate from this repository. Public
feature ideas, formulas and parameter starting points are attributed above.
No complete third-party notebook is redistributed as a local module. This project
adapts methods rather than presenting them as independently discovered techniques.

Actual downloaded source hashes and API version numbers are recorded below.

- `kirill0212/s6e9-lightgbm`: version 2, source SHA-256 `365bf64ed8db622f1f865381971011012b811c685de1b49e91d3cfcc16e4ca42`.

- `mikhailnaumov/electric-vehicle-purchases-single-xgb`: version 4, source SHA-256 `990f29ae8d148b4fa3e813d8013ed32f0e8ecf46c75684e5011ac32419c4b789`.

- `jazivxt/single-model-zoom-zoom`: version 9, source SHA-256 `3051f245472ff283297f654a611a437cbbd3e8ff20f8eab7e94c17e5ace0dec6`.

- `miickey/0-94637-lb-multi-scale-rank-transfer-blend`: version 9, source SHA-256 `db73b3efb35814460268d0722717b999a4f012b67bf703a00831021c51d9ebf9`.

- `chovyxu/ev-adoption-tokens-xgboost-tinytokentransformer`: version 13, source SHA-256 `dcace943640ee3a6ff44d181b7a40c6c321423831d8f3e318f8a280820ba9711`.
