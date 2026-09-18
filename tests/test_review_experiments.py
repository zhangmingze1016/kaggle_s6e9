import unittest
import numpy as np
from training.review_experiments import compare


class ReviewTest(unittest.TestCase):
    def test_fold_deltas_and_identity(self):
        base = dict(y_true=np.array([0, 1, 0, 1]), fold_ids=np.array([0, 0, 1, 1]),
                    oof_pred=np.array([.1, .9, .2, .8]))
        same = compare(base, base)
        self.assertEqual(same['delta'], 0)
        self.assertEqual(same['mean_abs_difference'], 0)
        candidate = {**base, 'oof_pred':np.array([.9, .1, .2, .8])}
        result = compare(base, candidate)
        self.assertEqual([r['delta'] for r in result['folds']], [-1., 0.])
        self.assertEqual(result['positive_folds'], 0)

    def test_invalid_fold_rejected(self):
        base = dict(y_true=np.array([0, 1]), fold_ids=np.array([0., np.nan]),
                    oof_pred=np.array([.1, .9]))
        with self.assertRaises(ValueError):
            compare(base, base)
