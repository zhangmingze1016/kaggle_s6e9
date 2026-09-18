import unittest

from training.encoding_trial import trial_args
from utils.cv_runner import parse_args


class EncodingTrialTest(unittest.TestCase):
    def test_only_inner_cv_changes(self):
        baseline = vars(parse_args('LightGBM', ['--preset', 'strong']))
        candidate = vars(trial_args())
        differences = {key for key in baseline if baseline[key] != candidate[key]}
        self.assertEqual(differences, {'te_cv'})
        self.assertEqual(candidate['te_cv'], 10)
        self.assertIsNone(candidate['sample_size'])
        self.assertEqual(candidate['random_seed'], 42)
