import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from utils.feature_engineering import NotebookFeatureEngineer, TripleTargetEncoder


class NotebookFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.raw = pd.read_csv('data/train.csv', nrows=100)
        self.X = self.raw.drop(columns=['id', 'Will_Buy_EV'])

    def test_schema_and_unchanged_inputs(self):
        before = self.X.copy(deep=True)
        test = self.X.iloc[:10].copy()
        test['Gender'] = 'Unseen'
        tr, te = NotebookFeatureEngineer().transform_pair(self.X, test)
        assert_frame_equal(self.X, before)
        self.assertTrue(tr.columns.equals(te.columns))
        self.assertTrue(tr.index.equals(self.X.index))
        self.assertTrue(te.index.equals(test.index))
        self.assertFalse(tr.isna().any().any())
        self.assertTrue(set(NotebookFeatureEngineer.NUMERIC_COLUMNS).issubset(tr.columns))
        self.assertFalse(any(tr.select_dtypes(include=['object', 'string']).columns))

    def test_fe_b_values(self):
        base = NotebookFeatureEngineer._base_features(self.X)
        np.testing.assert_array_equal(base['total_charging'],
            self.X.Charging_Stations_Near_Home + self.X.Charging_Stations_Near_Work)
        np.testing.assert_allclose(base['income_per_car'],
            self.X.Annual_Income_USD / (self.X.Number_of_Cars_Owned + 1))

    def test_no_self_target_and_unseen_fallback(self):
        # Unique values cannot encode their own label in cross-fitted training.
        n = 100
        X = pd.DataFrame({c: np.arange(n) for c in NotebookFeatureEngineer.NUMERIC_COLUMNS})
        y = np.tile([0, 1], n // 2)
        encoder = TripleTargetEncoder()
        encoded = encoder.fit_transform(X, y).filter(like='TE_')
        np.testing.assert_allclose(encoded, .5)
        unseen = X.iloc[:3].copy() + 1000
        transformed = encoder.transform(unseen).filter(like='TE_')
        np.testing.assert_allclose(transformed, .5)
        self.assertEqual(encoded.shape[1], 21)

    def test_validation_transform_does_not_update_maps(self):
        encoder = TripleTargetEncoder()
        y = self.raw.Will_Buy_EV.eq('Yes').astype(int)
        encoder.fit_transform(self.X, y)
        before = encoder.transform(self.X.iloc[:5])
        encoder.transform(self.X.iloc[10:20])
        assert_frame_equal(before, encoder.transform(self.X.iloc[:5]))

    def test_target_rejected(self):
        with self.assertRaises(ValueError):
            NotebookFeatureEngineer().transform_pair(self.raw, self.raw)


if __name__ == '__main__':
    unittest.main()
