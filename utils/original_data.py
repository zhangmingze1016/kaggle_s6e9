"""Target statistics learned exclusively from the public original EV dataset."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_URL = 'https://www.kaggle.com/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety'
DOWNLOAD_URL = 'https://www.kaggle.com/api/v1/datasets/download/itzzomkar/ev-adoption-behavior-and-range-anxiety'
SOURCE_FILENAME = 'EV_Adoption_and_Range_Anxiety_Dataset.csv'
SOURCE_SHA256 = 'c271a380df51b18177d0a039d54525ca7c3d71500701dc7496714a091df16fae'
DEFAULT_PATH = 'data/external/ev_adoption.csv'
TARGET = 'Will_Buy_EV'
NUMERIC = ('Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
           'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Environmental_Concern_Level')
CATEGORICAL = ('Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible',
               'Subsidy_Available', 'Range_Anxiety_Level')
FEATURES = NUMERIC + CATEGORICAL


def canonical_features(frame):
    """Stable comparison across CSV dtype/column ordering; never includes IDs/labels."""
    result = frame.loc[:, list(FEATURES)].copy()
    for col in NUMERIC:
        result[col] = pd.to_numeric(result[col], errors='raise').astype('float64')
        if np.isinf(result[col].to_numpy()).any():
            raise ValueError(f'Infinite source feature: {col}')
        # Normalize negative zero for row hashing.
        result.loc[result[col] == 0, col] = 0.
    for col in CATEGORICAL:
        result[col] = result[col].astype('string').fillna('__MISSING__')
    return result


def feature_hashes(frame):
    return pd.util.hash_pandas_object(canonical_features(frame), index=False)


def load_source(path=DEFAULT_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'{path}: first run python -m training.prepare_original_data')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError('Original dataset checksum differs from verified version 1; re-audit before use')
    return pd.read_csv(path)


class OriginalTargetStatistics:
    """13 smoothed means from disjoint external labels; competition y is not accepted.

    Remove external feature rows matching ANY competition train/test row before
    fitting. This conservative global exclusion also protects sampled CV runs.
    Unknown/missing keys fall back to the retained external target prior.
    """
    def __init__(self, smooth=20.):
        if not np.isfinite(smooth) or smooth <= 0:
            raise ValueError('smooth must be positive and finite')
        self.smooth = float(smooth)

    def fit(self, source, competition_frames):
        if not source[TARGET].isin(['Yes', 'No']).all():
            raise ValueError('Original data must have valid Yes/No labels')
        source = source.reset_index(drop=True)
        hashes = feature_hashes(source)
        excluded = np.zeros(len(source), dtype=bool)
        overlaps = []
        for frame in competition_frames:
            matched = hashes.isin(feature_hashes(frame)).to_numpy()
            overlaps.append(int(matched.sum()))
            excluded |= matched
        # Drop all ambiguous duplicate feature groups, not an arbitrary label.
        duplicated = hashes.duplicated(keep=False).to_numpy()
        kept = source.loc[~(excluded | duplicated)].copy()
        if len(kept) == 0 or kept[TARGET].nunique() != 2:
            raise ValueError('External rows remaining after exclusions must contain both classes')
        values = canonical_features(kept)
        labels = kept[TARGET].eq('Yes').astype(float)
        self.prior_ = float(labels.mean())
        self.maps_ = {}
        for col in FEATURES:
            # Missing external values do not define a reusable numeric/category key.
            valid = kept[col].notna()
            stats = pd.DataFrame({'key':values.loc[valid, col], 'target':labels.loc[valid]}).groupby('key').target.agg(['sum', 'count'])
            self.maps_[col] = (stats['sum'] + self.smooth * self.prior_) / (stats['count'] + self.smooth)
        self.audit_ = dict(source_rows=len(source), retained_rows=len(kept),
            matching_rows_by_competition_frame=overlaps, excluded_matching_rows=int(excluded.sum()),
            duplicate_feature_rows=int(duplicated.sum()), source_positive_rate=float(source[TARGET].eq('Yes').mean()),
            retained_positive_rate=self.prior_, smooth=self.smooth, features=len(FEATURES),
            source=SOURCE_URL, source_sha256=SOURCE_SHA256,
            note='Competition labels are never used. Exact overlaps and external duplicates are excluded; near duplicates are not ruled out.')
        return self

    def transform(self, frame):
        values = canonical_features(frame)
        return pd.DataFrame({f'original_TE_{col}':values[col].map(self.maps_[col]).fillna(self.prior_).astype('float32')
                             for col in FEATURES}, index=frame.index)

    def coverage(self, frame):
        values = canonical_features(frame)
        return {col:float(values[col].isin(self.maps_[col].index).mean()) for col in FEATURES}
