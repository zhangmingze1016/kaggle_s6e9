import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from utils.cv_runner import parse_args
from utils.original_data import OriginalTargetStatistics, FEATURES, NUMERIC, TARGET, feature_hashes, load_source


def source_fixture():
    source = pd.read_csv('data/train.csv', nrows=12)
    source[TARGET] = ['Yes','No'] * 6
    source['Age'] = np.arange(30,42)
    return source


class OriginalDataTest(unittest.TestCase):
    def test_overlap_excluded_and_competition_labels_ignored(self):
        source = source_fixture()
        # Dtype differences and arbitrary frame indices must not hide duplicates.
        comp = source.iloc[[0]].copy()
        comp['Age'] = comp.Age.astype(float)
        comp.index = [88]
        first = OriginalTargetStatistics().fit(source, [comp])
        self.assertEqual(first.audit_['excluded_matching_rows'],1)
        self.assertEqual(first.audit_['retained_rows'],11)
        changed = source.copy(); changed.loc[0,TARGET] = 'No'
        comp[TARGET] = 'invalid-competition-label-unused'
        second = OriginalTargetStatistics().fit(changed, [comp])
        assert_frame_equal(first.transform(source), second.transform(source))
        self.assertEqual(first.transform(comp).index.tolist(), [88])
        self.assertAlmostEqual(float(first.transform(comp)['original_TE_Age'].iloc[0]),first.prior_,places=6)

    def test_smoothing_unknowns_and_missing(self):
        source = source_fixture()
        encoder = OriginalTargetStatistics(smooth=20).fit(source, [])
        values = encoder.transform(source)
        self.assertEqual(values.shape,(12,13))
        self.assertAlmostEqual(float(values.original_TE_Age.iloc[0]),(1+20*.5)/21,places=6)
        unseen = source.iloc[[0]].copy()
        for col in FEATURES:
            unseen[col] = np.nan if col in NUMERIC else 'unknown'
        np.testing.assert_allclose(encoder.transform(unseen),.5)

    def test_conflicting_duplicates_removed(self):
        source=source_fixture()
        duplicate=source.iloc[[0]].copy();duplicate[TARGET]='No'
        encoder=OriginalTargetStatistics().fit(pd.concat([source,duplicate]),[])
        self.assertEqual(encoder.audit_['duplicate_feature_rows'],2)
        self.assertEqual(encoder.audit_['retained_rows'],11)

    def test_source_validation_and_opt_in(self):
        source=source_fixture()
        with self.assertRaises(ValueError):
            OriginalTargetStatistics().fit(source,[source])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.csv';path.write_text('wrong data')
            with self.assertRaisesRegex(ValueError,'checksum'):
                load_source(path)
        self.assertIsNone(parse_args('LightGBM',['--preset','strong']).original_data)
        args=parse_args('LightGBM',['--preset','strong','--original-data'])
        self.assertEqual(args.original_data,'data/external/ev_adoption.csv')
        self.assertEqual((args.n_splits,args.random_seed,args.model_seed),(5,42,42))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args('LightGBM',['--original-data'])

    def test_hash_ignores_ids_labels_and_column_order(self):
        source=source_fixture()
        changed=source[source.columns[::-1]].copy()
        changed['id']+=100;changed[TARGET]='unread'
        np.testing.assert_array_equal(feature_hashes(source),feature_hashes(changed))
