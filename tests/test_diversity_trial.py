import unittest
import numpy as np
from training.diversity_trial import trial_args, fixed_blend


class DiversityTrialTest(unittest.TestCase):
    def test_historical_cv_and_successful_features_preserved(self):
        args = trial_args()
        self.assertEqual((args.n_splits, args.random_seed, args.model_seed, args.te_cv),
                         (5, 42, 42, 5))
        self.assertEqual(args.feature_recipe, 'multiscale')
        self.assertTrue(args.target_encoding and args.income_neighbors)
        self.assertFalse(args.local_windows)
        self.assertIsNone(args.original_data)
        self.assertIsNone(args.sample_size)
        self.assertEqual(args.depth, 7)

    def test_fixed_blend_does_not_mutate_baseline(self):
        baseline = dict(oof_pred=np.array([.1, .7]), test_pred=np.array([.4]))
        candidate = dict(oof_pred=np.array([.3, .9]), test_pred=np.array([.8]))
        blended = fixed_blend(baseline, candidate)
        np.testing.assert_allclose(blended['oof_pred'], [.14, .74])
        np.testing.assert_allclose(blended['test_pred'], [.48])
        np.testing.assert_allclose(baseline['oof_pred'], [.1, .7])
