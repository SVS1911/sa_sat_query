import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from controller.input_validator import validate
from utils.image_io import LoadedImage


class InputValidatorTests(unittest.TestCase):
    def test_visual_pair_with_different_sizes_is_accepted(self):
        a = LoadedImage(np.zeros((20, 30, 3)), "a.png", bands=3, height=20, width=30,
                        modality_guess="optical")
        b = LoadedImage(np.zeros((24, 35, 3)), "b.png", bands=3, height=24, width=35,
                        modality_guess="optical")
        result = validate([a, b])
        self.assertTrue(result.ok)
        self.assertEqual(result.scenario, "bi_temporal")


if __name__ == "__main__":
    unittest.main()
