# Fold difficulty investigation

Seed 42 remains the primary benchmark. No production training code or configuration was changed.

## Findings

The current runner and historical CatBoost/XGBoost code use 5-fold shuffled StratifiedKFold with seed 42. The loader preserves CSV row order; IDs are excluded from model features. Same seed alone is not enough: identical ordered labels, row set and fold count are required. The preserved reference LightGBM used 10 folds, so it is not the same partition.

Nine completed checkpoint sets (45 folds) and one incomplete set (one fold) matched their reconstructed seed-42 indices exactly. All ten available full-data OOF bundles matched the primary fold assignments after ID alignment. The early historical model outputs have no surviving fold-index artifacts, so their equality is supported by source inspection, not directly proven from saved assignments.

All primary folds contain 133,733 rows. Fold 1's positive rate is 0.174638997; other folds are approximately 0.174646475 (one positive row difference). All input feature missing rates are zero. IDs equal row positions, but every fold spans almost the entire ID range. Fold 1 versus Fold 3 has at most about 0.0063 absolute standardized numerical mean difference overall, and 0.0134 within target classes. Largest categorical total variation is 0.00380 (home charging); ID-decile total variation is 0.00262. These small marginal differences do not establish a causal explanation for difficulty.

No exact duplicate feature rows, contradictory labels on exact feature duplicates, or exact train/test feature matches were found using row hashes. This does not exclude near duplicates, latent groups, or synthetic-generator effects. Only train/test/sample-submission data are present; original-data relationships cannot be established without the source data. Detailed category combinations, repeated-income cohorts, ID buckets, numerical quantiles and class-conditional contrasts are exported for inspection.

## Alternate-seed probe

A bounded LightGBM diagnostic trained 150 trees per fold on raw features, depth 4, 15 leaves, learning rate 0.08, fixed model seed 42, without target encoding or early stopping. This tests actual held-out performance under alternative partitions; it does not reassign existing OOF predictions and call that new CV. It is not a full rerun of the best pipeline and must not be used to choose a favorable benchmark seed.

| Split seed | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---:|---:|---:|---:|---:|
| 42 | 0.9395703 | 0.9407830 | 0.9420408 | 0.9415423 | 0.9408311 |
| 123 | 0.9410612 | 0.9415651 | 0.9397899 | 0.9410640 | 0.9411175 |
| 2026 | 0.9403356 | 0.9400998 | 0.9422772 | 0.9408977 | 0.9408233 |
| 3407 | 0.9412734 | 0.9413082 | 0.9413413 | 0.9398480 | 0.9404720 |
| 8888 | 0.9410626 | 0.9404303 | 0.9405007 | 0.9413751 | 0.9412390 |

The original pattern is not persistent across seeds. Fold numbers are arbitrary partition labels, not a time sequence. Across correlated models using the same held-out rows, it is expected that difficult rows remain difficult; these model runs are not independent repetitions of the split experiment.

## Model improvements

Relative to v20, v27 improves all five folds (roughly +0.000025, +0.000043, +0.000009, +0.000013, +0.000020). Its improvement is not driven by Fold 3. v28 loses on all five folds versus v20. v30 (income-neighborhood removal) loses on four folds and improves only Fold 4; the available OOF evidence does not support removing that feature. v29 improves four of five folds relative to v27, but pooled gain is only about 0.000005 and follows repeated blend selection on the same OOF data.

## Interpretation and limits

The evidence is consistent with ordinary realized sampling variation in a fixed partition, with shared case difficulty across models. There is no demonstrated meaningful ID, missingness, or exact-duplicate structure explaining the pattern. This is not proof that the synthetic dataset lacks hidden structure, nor proof that Fold 1 is intrinsically harder under every model. Marginal summaries cannot fully explain joint feature-label overlap. Exported conditional AUC standard errors describe evaluation-row uncertainty for fixed predictions only; overlapping training sets and prior model selection prevent treating them as independent CV significance tests.

Do not change seed 42, optimize specifically for Fold 3, or choose seeds by their reported scores. Evaluate changes using paired per-fold deltas plus pooled OOF; use alternate seeds only as robustness checks.

## Reproduce

```bash
python -m training.diagnose_folds
python -m training.diagnose_folds --probe
```

The first command only analyzes saved data/artifacts. The second also trains 25 bounded diagnostic models. All output goes to `artifacts/fold_diagnostics/`; no submission CSV or experiment version is created. See `audit.json`, `numeric.csv`, `categorical.csv`, `missing.csv`, `structural_cohorts.csv`, `model_fold_delta_vs_v020.csv`, and `alternate_seed_probe.csv`. Probe results/configuration are retained when a subsequent diagnostics-only run is made.

Reference: [scikit-learn StratifiedKFold documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html).

For v27, Fold 1 negative-score 90th percentile is 0.31492 versus 0.30662 in Fold 3; positive-score 10th percentile is 0.20035 versus 0.20502. This shows slightly greater class-score overlap in Fold 1, a description of observed difficulty rather than its causal source. Conditional AUC standard errors are approximately 0.00065–0.00066 per fold. Target lag-1 correlation by original row order is only 0.000987.
