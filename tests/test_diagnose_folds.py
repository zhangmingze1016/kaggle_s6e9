import unittest
import numpy as np
from training.diagnose_folds import assignments


class FoldDiagnosticTest(unittest.TestCase):
    def test_reproducible_balanced_and_seed_sensitive(self):
        y=np.tile([0,0,0,1],100)
        a=assignments(y,42)
        np.testing.assert_array_equal(a,assignments(y,42))
        self.assertFalse(np.array_equal(a,assignments(y,123)))
        self.assertEqual(set(a),set(range(5)))
        for fold in range(5):
            self.assertEqual((a==fold).sum(),80)
            self.assertEqual(y[a==fold].mean(),.25)

    def test_row_order_matters(self):
        y=np.tile([0,1],100)
        permutation=np.random.default_rng(4).permutation(len(y))
        original=assignments(y,42)
        restored=np.empty_like(original)
        restored[permutation]=assignments(y[permutation],42)
        self.assertFalse(np.array_equal(original,restored))
