import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.image_io import load_image
from utils.spectral_indices import classify_landcover


class ImageNormalizationTests(unittest.TestCase):
    def test_rgb_upload_keeps_green_as_green(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "forest.png")
            Image.fromarray(np.full((20, 20, 3), [20, 90, 25], dtype=np.uint8)).save(path)
            loaded = load_image(path)
            self.assertLess(float(loaded.array[..., 0].mean()), float(loaded.array[..., 1].mean()))
            self.assertLess(classify_landcover(loaded.array)[1]["water"], 0.05)


if __name__ == "__main__":
    unittest.main()
