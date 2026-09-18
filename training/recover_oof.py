"""Restore a completed run's OOF bundle without fitting any model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def recover(run_dir, artifact_dir='artifacts/predictions'):
    root = Path(run_dir)
    config = json.loads((root / 'config.json').read_text())
    result = json.loads((root / 'result.json').read_text())
    if config.get('sample_size') is not None:
        raise ValueError('Recovery currently supports full-data runs only')
    for path, expected in config['data_hashes'].items():
        with open(path, 'rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'Data changed: {path}')
    train = pd.read_csv(config['train_path'], usecols=['id', 'Will_Buy_EV'])
    test = pd.read_csv(config['test_path'], usecols=['id'])
    if not train.Will_Buy_EV.isin(['Yes', 'No']).all():
        raise ValueError('Unexpected target labels')
    y = train.Will_Buy_EV.eq('Yes').astype(int).to_numpy()
    oof = np.full(len(train), np.nan)
    folds = np.full(len(train), -1, dtype=np.int16)
    prediction = np.zeros(len(test))
    cv = StratifiedKFold(config['n_splits'], shuffle=True, random_state=config['random_seed'])
    for fold, (_, valid) in enumerate(cv.split(train, y)):
        with np.load(root / f'fold_{fold+1:02d}.npz', allow_pickle=False) as data:
            if not np.array_equal(valid, data['valid_indices']):
                raise ValueError('Checkpoint split mismatch')
            for key, size in [('valid_pred', len(valid)), ('test_pred', len(test))]:
                values = data[key]
                if values.shape != (size,) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                    raise ValueError(f'Invalid checkpoint {key}')
            oof[valid] = data['valid_pred']
            prediction += data['test_pred'] / config['n_splits']
            folds[valid] = fold
    submission = pd.read_csv(result['submission']).set_index('id')
    if not submission.index.is_unique or set(submission.index) != set(test.id):
        raise ValueError('Submission IDs differ')
    np.testing.assert_allclose(prediction, submission.loc[test.id, 'Will_Buy_EV'], rtol=0, atol=1e-12)
    target = Path(artifact_dir) / Path(result['submission']).with_suffix('.npz').name
    target.parent.mkdir(parents=True, exist_ok=True)
    bundle = dict(train_ids=train.id.to_numpy(), test_ids=test.id.to_numpy(),
                  y_true=y, fold_ids=folds, oof_pred=oof, test_pred=prediction)
    if target.exists():
        with np.load(target, allow_pickle=False) as existing:
            for key, values in bundle.items():
                np.testing.assert_array_equal(existing[key], values)
    else:
        temporary = target.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, **bundle)
        temporary.replace(target)
    print(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('--artifact-dir', default='artifacts/predictions')
    args = parser.parse_args()
    recover(args.run_dir, args.artifact_dir)


if __name__ == '__main__':
    main()
