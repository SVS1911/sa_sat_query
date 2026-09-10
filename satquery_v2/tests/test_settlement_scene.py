import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.spectral_indices import classify_landcover


class SettlementSceneTests(unittest.TestCase):
    def test_bright_neutral_settlement_is_not_all_bare_ground(self):
        image = np.zeros((60, 100, 3), dtype=np.float32)
        image[:] = (0.04, 0.16, 0.05)  # vegetation
        image[20:45, 10:90] = (0.55, 0.48, 0.42)  # settlement and roads
        _, proportions, _ = classify_landcover(image)
        self.assertGreater(proportions["built_up"], 0.15)


if __name__ == "__main__":
    unittest.main()
