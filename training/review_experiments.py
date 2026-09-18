"""Review existing OOF predictions without training or selecting new weights."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils.ensemble import load_bundles


def compare(baseline, candidate):
    y, folds = baseline['y_true'], baseline['fold_ids']
    if not np.isfinite(folds).all() or not np.equal(folds, np.floor(folds)).all():
        raise ValueError('Fold IDs must be finite integers')
    rows = []
    for fold in np.unique(folds):
        mask = folds == fold
        if np.unique(y[mask]).size != 2:
            raise ValueError(f'Fold {fold} does not contain both classes')
        reference = float(roc_auc_score(y[mask], baseline['oof_pred'][mask]))
        score = float(roc_auc_score(y[mask], candidate['oof_pred'][mask]))
        rows.append(dict(fold=int(fold)+1, rows=int(mask.sum()), baseline_auc=reference,
                         auc=score, delta=score-reference))
    reference = float(roc_auc_score(y, baseline['oof_pred']))
    score = float(roc_auc_score(y, candidate['oof_pred']))
    a, b = baseline['oof_pred'], candidate['oof_pred']
    correlation = float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else None
    rank_correlation = float(pd.Series(a).rank().corr(pd.Series(b).rank())) if correlation is not None else None
    return dict(oof_auc=score, delta=score-reference,
                positive_folds=sum(r['delta'] > 1e-12 for r in rows),
                total_folds=len(rows), min_fold_delta=min(r['delta'] for r in rows),
                max_fold_delta=max(r['delta'] for r in rows),
                correlation=correlation, rank_correlation=rank_correlation, mean_abs_difference=float(np.mean(np.abs(a-b))),
                folds=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='v027')
    parser.add_argument('--artifact-dir', default='artifacts/predictions')
    parser.add_argument('--report-dir', default='artifacts/reviews')
    args = parser.parse_args(argv)
    root, output = Path(args.artifact_dir), Path(args.report_dir)
    if output.resolve().is_relative_to(Path('predictions').resolve()):
        parser.error('Reports must be outside predictions/')
    matches = list(root.glob(f'prediction_{args.baseline}_*.npz'))
    if len(matches) != 1:
        parser.error(f'Expected one baseline bundle for {args.baseline}; found {len(matches)}')
    baseline = matches[0]
    results, skipped = [], []
    for path in sorted(root.glob('prediction_v*.npz')):
        try:
            ref, candidate = load_bundles([baseline, path])
            result = compare(ref, candidate)
        except (ValueError, KeyError, OSError) as exc:
            if path == baseline:
                raise
            skipped.append(dict(file=str(path), reason=str(exc)))
            continue
        results.append(dict(version=path.name.split('_')[1], file=str(path), **result))
    if not results:
        raise ValueError('No comparable bundles')
    output.mkdir(parents=True, exist_ok=True)
    note = ('Retrospective OOF diagnostics only. Reused folds and previously selected blends '
            'are not independent validation; fold wins are not a significance test. '
            'Probability differences are not directly comparable to rank-blend differences. '
            'No training, new weight selection, or submission generation is performed.')
    report = dict(baseline=str(baseline), note=note, models=results, skipped=skipped)
    (output/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    summary = pd.DataFrame([{k:v for k,v in r.items() if k not in ('folds','file')} for r in results])
    summary.to_csv(output/'summary.csv', index=False)
    pd.DataFrame([dict(version=r['version'], **fold) for r in results for fold in r['folds']]).to_csv(output/'folds.csv', index=False)
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.8f}'))
    print(f'\nReports: {output}; skipped incompatible bundles: {len(skipped)}')
    print(note)


if __name__ == '__main__':
    main()
