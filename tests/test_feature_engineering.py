import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from utils.data_loader import EVDataLoader
from utils.feature_engineering import EVFeatureEngineer
from utils.prediction_saver import PredictionSaver


class EVFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.raw = pd.read_csv('data/train.csv', nrows=12)
        self.X = self.raw.drop(columns=['id', 'Will_Buy_EV'])
        self.engineer = EVFeatureEngineer()

    def test_row_independence_and_input_unchanged(self):
        original = self.X.copy(deep=True)
        result = self.engineer.transform(self.X)
        separate = pd.concat([self.engineer.transform(self.X.iloc[:5]),
                              self.engineer.transform(self.X.iloc[5:])])
        assert_frame_equal(result, separate)
        assert_frame_equal(self.X, original)
        self.assertEqual(result.shape[1], 34)
        self.assertNotIn('Will_Buy_EV', result)

    def test_zero_missing_and_unknown(self):
        self.X.loc[0, ['Charging_Stations_Near_Home',
                       'Charging_Stations_Near_Work', 'Number_of_Cars_Owned']] = 0
        self.X.loc[0, 'Home_Charging_Possible'] = 'No'
        self.X.loc[1, 'Range_Anxiety_Level'] = 'Unknown'
        self.X.loc[2, 'City_Type'] = None
        self.X.loc[3, 'Charging_Stations_Near_Home'] = np.nan
        result = self.engineer.transform(self.X)
        self.assertEqual(result.loc[0, 'No_Charging_Access'], 1)
        self.assertTrue(pd.isna(result.loc[0, 'Income_Per_Car']))
        self.assertTrue(pd.isna(result.loc[0, 'Commute_Per_Charging_Station']))
        self.assertTrue(pd.isna(result.loc[1, 'Range_Anxiety_Ordinal']))
        self.assertEqual(result.loc[2, 'City_Type'], '__MISSING__')
        self.assertTrue(pd.isna(result.loc[3, 'No_Public_Charging']))
        self.assertFalse(np.isinf(result.select_dtypes('number')).any().any())

    def test_invalid_input(self):
        for invalid in [self.raw, self.X.drop(columns='Age'),
                        self.engineer.transform(self.X)]:
            with self.assertRaises(ValueError):
                self.engineer.transform(invalid)

    def test_loader_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            train, test = Path(tmp) / 'train.csv', Path(tmp) / 'test.csv'
            self.raw.to_csv(train, index=False)
            self.raw.drop(columns='Will_Buy_EV').to_csv(test, index=False)
            baseline = EVDataLoader(train, test).load()
            X, y, Xt, ids, cats = EVDataLoader(train, test, feature_engineering=True).load()
            assert_frame_equal(X, Xt)
            self.assertEqual(len(baseline[0].columns), 13)
            pd.testing.assert_series_equal(y, baseline[1])
            pd.testing.assert_series_equal(ids, baseline[3])
            self.assertEqual(len(cats), 9)

    def test_experiment_schema_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'experiments.csv'
            pd.DataFrame([{'version': 'v000', 'model': 'CatBoost', 'oof_auc': .9}]).to_csv(log, index=False)
            saver = PredictionSaver(Path(tmp) / 'predictions', log)
            path = saver.save([1], [.5], .91, {'model': 'CatBoost', 'feature_engineering': True})
            self.assertIn('_fe_', path.name)
            records = pd.read_csv(log)
            self.assertEqual(len(records), 2)
            self.assertEqual(records.loc[1, 'oof_auc'], .91)
            self.assertEqual(records.loc[1, 'feature_engineering'], True)


if __name__ == '__main__':
    unittest.main()
