"""Full-data inner-CV-10 encoding trial; keep the primary outer CV unchanged."""
import argparse
import json
from pathlib import Path

from training.review_experiments import compare
from utils.cv_runner import parse_args, run_cv
from utils.ensemble import load_bundles


def trial_args(n_jobs=4):
    return parse_args('LightGBM', [
        '--preset', 'strong', '--te-cv', '10', '--n-splits', '5',
        '--random-seed', '42', '--model-seed', '42', '--n-jobs', str(n_jobs),
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-jobs', type=int, default=4)
    args = parser.parse_args(argv)
    root = Path('artifacts/predictions')
    references = {}
    for version in ('v020', 'v027'):
        matches = list(root.glob(f'prediction_{version}_*.npz'))
        if len(matches) != 1:
            parser.error(f'Expected one {version} reference bundle; found {len(matches)}')
        references[version] = matches[0]
    # Fail before training if historical reference IDs, labels or folds differ.
    load_bundles(list(references.values()))
    result = run_cv('LightGBM', trial_args(args.n_jobs))
    comparisons = {}
    for version, path in references.items():
        baseline, candidate = load_bundles([path, result['bundle']])
        comparisons[version] = compare(baseline, candidate)
    report = dict(
        candidate=result['bundle'], references={k: str(v) for k, v in references.items()},
        comparisons=comparisons,
        note='Only inner encoding CV changes from 5 to 10, including income neighbors. '
             'Outer five-fold seed 42 and model seed 42 stay fixed. Retrospective OOF '
             'diagnostics, not independent validation or proof of leaderboard improvement. '
             'No blend search or automatic replacement of the baseline.',
    )
    output = Path('artifacts/encoding_trial') / result['params']['run_signature']
    output.mkdir(parents=True, exist_ok=True)
    (output / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    for version, comparison in comparisons.items():
        print(f"Against {version}: delta={comparison['delta']:+.8f}; "
              f"positive folds={comparison['positive_folds']}/{comparison['total_folds']}")
    print(f'Report: {output / "comparison.json"}')


if __name__ == '__main__':
    main()
