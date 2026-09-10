import csv
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from controller.agentic_controller import SatQueryController
from models.captioning import CaptionModel, load_caption_records


class CaptioningTests(unittest.TestCase):
    def _dataset(self, root):
        os.makedirs(os.path.join(root, "train"), exist_ok=True)
        image_path = os.path.join(root, "train", "airport_1.jpg")
        Image.fromarray(np.full((24, 24, 3), [190, 190, 190], dtype=np.uint8)).save(image_path)
        with open(os.path.join(root, "train.csv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["captions", "filepath"])
            writer.writerow([r"""['planes near a runway.' 'aircraft at an airport.']""", "train/airport_1.jpg"])
        return image_path

    def test_csv_parsing_and_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            image_path = self._dataset(root)
            records = load_caption_records(root, split="train")
            self.assertEqual(records[0].image_path, image_path)
            self.assertEqual(len(records[0].captions), 2)
            model = CaptionModel.train(records, seed=3)
            result = model.predict(image_path)
            self.assertIn(result["caption"], records[0].captions)
            checkpoint = os.path.join(root, "captioning.json")
            model.save(checkpoint)
            self.assertEqual(CaptionModel.load(checkpoint).predict(image_path)["caption"], result["caption"])

    def test_controller_uses_dataset_caption_backend(self):
        with tempfile.TemporaryDirectory() as root:
            image_path = self._dataset(root)
            model = CaptionModel.train(load_caption_records(root), seed=0)
            checkpoint = os.path.join(root, "captioning.json")
            model.save(checkpoint)
            controller = SatQueryController(reports_dir=os.path.join(root, "reports"),
                                            caption_checkpoint=checkpoint)
            result = controller.run([image_path], "Describe this image.")
            self.assertTrue(result.success)
            self.assertTrue(result.answer.startswith("This image appears to show: "))
            # The answer is now a structured block — headline, then measured
            # values, then provenance — so the caption is contained in it
            # rather than terminating it.
            self.assertTrue(
                any(phrase in result.answer
                    for phrase in ("planes near a runway.", "aircraft at an airport."))
            )
            self.assertIn("Source:", result.answer)
            self.assertTrue(any("dataset-nearest-neighbor" in line for line in result.audit_trail))


if __name__ == "__main__":
    unittest.main()
