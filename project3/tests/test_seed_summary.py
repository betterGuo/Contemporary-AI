"""The interaction is a difference of simple effects; paired switches explain the net gain."""
import unittest
import numpy as np
from summarize_seeds import directions, effect_sizes, spread, switches


class SeedSummaryTests(unittest.TestCase):
    def test_positive_gain_can_have_negative_interaction(self):
        # Augmentation adds 10 without attention, 4 with attention.
        result = effect_sizes(dict(baseline=60, augment=70, attention=68, both=72),100)
        self.assertEqual(result, dict(augmentation_main=7., attention_main=5., interaction=-6.))

    def test_switches_preserve_both_directions(self):
        a=np.array([False,False,True,True,True]); b=np.array([True,True,False,True,True])
        self.assertEqual(switches(a,b),dict(corrected=2,regressed=1,both_correct=2,both_wrong=0))
        self.assertEqual(spread([1,2,3]),dict(mean=2.,sample_sd=1.))
        self.assertEqual(directions([-1,0,2]),dict(positive=1,zero=1,negative=1))
