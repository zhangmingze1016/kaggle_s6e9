"""Test CatBoost with the successful feature view and one fixed blend."""
import argparse
import json
from pathlib import Path

from training.review_experiments import compare
from utils.cv_runner import parse_args, run_cv
from utils.ensemble import load_bundles


def trial_args(n_jobs=4):
    return parse_args('CatBoost', [
        '--preset', 'strong', '--depth', '7', '--iterations', '5000',
        '--learning-rate', '.03', '--l2-leaf-reg', '10', '--rsm', '.8',
        '--early-stopping-rounds', '250', '--n-splits', '5',
        '--random-seed', '42', '--model-seed', '42', '--te-cv', '5',
        '--n-jobs', str(n_jobs),
    ])


def fixed_blend(baseline, candidate):
    """Inputs must already be ID-aligned and fold-validated by load_bundles."""
    return {**baseline,
            'oof_pred': .8 * baseline['oof_pred'] + .2 * candidate['oof_pred'],
            'test_pred': .8 * baseline['test_pred'] + .2 * candidate['test_pred']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-jobs', type=int, default=4)
    args = parser.parse_args(argv)
    matches = list(Path('artifacts/predictions').glob('prediction_v027_*.npz'))
    if len(matches) != 1:
        parser.error(f'Expected one v027 baseline bundle; found {len(matches)}')
    reference = matches[0]
    load_bundles([reference, reference])  # Validate before expensive training.
    result = run_cv('CatBoost', trial_args(args.n_jobs))
    baseline, candidate = load_bundles([reference, result['bundle']])
    report = dict(
        baseline=str(reference), candidate=result['bundle'],
        candidate_comparison=compare(baseline, candidate),
        fixed_blend_weights=[.8, .2],
        fixed_blend_comparison=compare(baseline, fixed_blend(baseline, candidate)),
        note='Same outer five folds, seed 42. Blend weight fixed before training; '
             'no search, automatic promotion or blend submission. Historical OOF '
             'has been reused and is not independent confirmation of improvement.',
    )
    output = Path('artifacts/diversity_trial') / result['params']['run_signature']
    output.mkdir(parents=True, exist_ok=True)
    (output / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    for key in ('candidate_comparison', 'fixed_blend_comparison'):
        row = report[key]
        print(f"{key}: AUC={row['oof_auc']:.8f}; delta={row['delta']:+.8f}; "
              f"positive folds={row['positive_folds']}/{row['total_folds']}")
    print(f'Report: {output / "comparison.json"}')


if __name__ == '__main__':
    main()
