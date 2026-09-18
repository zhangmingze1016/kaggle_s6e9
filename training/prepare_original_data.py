"""Download the pinned public source dataset and audit overlap/coverage, without training."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

from utils.original_data import (DEFAULT_PATH, DOWNLOAD_URL, SOURCE_FILENAME, SOURCE_SHA256,
                                 OriginalTargetStatistics, load_source)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-data', default=DEFAULT_PATH)
    parser.add_argument('--train-path', default='data/train.csv')
    parser.add_argument('--test-path', default='data/test.csv')
    args = parser.parse_args(argv)
    path = Path(args.original_data)
    if path.resolve().is_relative_to(Path('predictions').resolve()):
        parser.error('External data must be outside predictions/')
    if not path.exists():
        with urlopen(DOWNLOAD_URL, timeout=60) as response:
            archive = response.read()
        with ZipFile(io.BytesIO(archive)) as zipped:
            content = zipped.read(SOURCE_FILENAME)
        if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
            raise ValueError('Downloaded source checksum changed; no file was written')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp.csv')
        temporary.write_bytes(content)
        temporary.replace(path)
    source = load_source(path)
    train, test = pd.read_csv(args.train_path), pd.read_csv(args.test_path)
    encoder = OriginalTargetStatistics().fit(source, [train, test])
    report = {**encoder.audit_, 'license':'CC0: Public Domain (Kaggle API metadata)',
              'rules_url':'https://www.kaggle.com/competitions/playground-series-s6e9/rules',
              'rules_checked':'2026-09-17; competition-specific section 2.6 permits publicly accessible external data',
              'train_coverage':encoder.coverage(train), 'test_coverage':encoder.coverage(test),
              'missing_rates':source.isna().mean().to_dict()}
    out = Path('artifacts/external_data'); out.mkdir(parents=True, exist_ok=True)
    (out/'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f'Source ready: {path}. No model was trained.')


if __name__ == '__main__':
    main()
