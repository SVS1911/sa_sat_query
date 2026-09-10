import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.change_analysis import compute_change_map


class ChangeAnalysisTests(unittest.TestCase):
    def test_different_sizes_are_resampled(self):
        before = np.zeros((20, 30, 3), dtype=np.float32)
        after = np.zeros((24, 35, 3), dtype=np.float32)
        after[8:16, 10:20] = 1.0
        result = compute_change_map(before, after)
        self.assertEqual(result["mask"].shape, after.shape[:2])
        self.assertGreater(result["changed_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
