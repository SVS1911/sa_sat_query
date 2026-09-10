import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.spectral_indices import classify_landcover


class IslandSceneTests(unittest.TestCase):
    def test_dark_forest_and_ocean_are_not_mostly_other(self):
        image = np.zeros((80, 80, 3), dtype=np.float32)
        image[:] = (0.03, 0.08, 0.22)  # dark ocean
        image[15:65, 15:65] = (0.04, 0.18, 0.06)  # dark green island
        _, proportions, _ = classify_landcover(image)
        self.assertGreater(proportions["water"], 0.25)
        self.assertGreater(proportions["vegetation"], 0.25)
        self.assertLess(proportions["other"], 0.20)


if __name__ == "__main__":
    unittest.main()
