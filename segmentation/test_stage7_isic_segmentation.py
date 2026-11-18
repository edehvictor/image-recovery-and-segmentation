"""Lightweight checks of pairing and split guards; no TensorFlow or training."""

import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import stage7_isic_segmentation as stage7


class ManifestTests(unittest.TestCase):
    def test_pairing_from_directories_and_missing_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()
            (images / "ISIC_0000001.jpg").write_bytes(b"image")
            (masks / "ISIC_0000001_segmentation.png").write_bytes(b"mask")
            pairs = stage7.pair_manifest(images, masks)
            self.assertEqual([p.image_id for p in pairs], ["ISIC_0000001"])
            (images / "ISIC_0000002.jpg").write_bytes(b"image")
            with self.assertRaisesRegex(ValueError, "do not match"):
                stage7.pair_manifest(images, masks)

    def test_pairing_from_nested_zips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images.zip"
            masks = root / "masks.zip"
            with ZipFile(images, "w") as archive:
                archive.writestr("ISIC2018_Task1-2_Training_Input/ISIC_0000010.jpg", b"image")
            with ZipFile(masks, "w") as archive:
                archive.writestr("ISIC2018_Task1_Training_GroundTruth/ISIC_0000010_segmentation.png", b"mask")
            pairs = stage7.pair_manifest(images, masks)
            self.assertEqual(len(pairs), 1)
            with stage7.PairReader(images, masks) as reader:
                self.assertEqual(reader.read(pairs[0]), (b"image", b"mask"))

    def test_duplicate_ids_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            (images / "a").mkdir()
            (images / "b").mkdir()
            (images / "a" / "ISIC_0000001.jpg").write_bytes(b"x")
            (images / "b" / "ISIC_0000001.jpg").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                stage7.source_members(images, stage7.IMAGE_RE)


class SplitTests(unittest.TestCase):
    def test_split_is_reproducible_disjoint_and_complete(self):
        ids = [f"ISIC_{i:07d}" for i in range(100)]
        first = stage7.make_split(ids, 42, 0.15, 0.15)
        second = stage7.make_split(reversed(ids), 42, 0.15, 0.15)
        self.assertEqual(first, second)
        self.assertEqual([len(first[k]) for k in ("train", "validation", "test")], [70, 15, 15])
        self.assertEqual(set().union(*map(set, first.values())), set(ids))
        self.assertFalse(set(first["train"]) & set(first["validation"]))
        self.assertFalse(set(first["train"]) & set(first["test"]))
        self.assertFalse(set(first["validation"]) & set(first["test"]))

    def test_unconfirmed_protocol_cannot_train(self):
        protocol = stage7.load_protocol(stage7.DEFAULT_PROTOCOL)
        # The returned experiment archive contains a confirmed protocol.
        # Exercise the draft guard on a copy without changing that record.
        protocol["confirmed_for_training"] = False
        self.assertEqual(protocol["losses"], ["BCE", "Dice"])
        with self.assertRaisesRegex(RuntimeError, "draft"):
            stage7.train_one(Path("missing-images"), Path("missing-masks"),
                             protocol, "Adam", "BCE", 0.001)


class PixelMetricTests(unittest.TestCase):
    def test_known_confusion_counts(self):
        scores = stage7.pixel_metrics(tp=3, fp=1, tn=4, fn=2)
        self.assertAlmostEqual(scores["precision"], 0.75)
        self.assertAlmostEqual(scores["accuracy"], 0.7)
        self.assertAlmostEqual(scores["specificity"], 0.8)
        self.assertAlmostEqual(scores["sensitivity"], 0.6)
        self.assertAlmostEqual(scores["dice"], 6 / 9)
        self.assertAlmostEqual(scores["iou"], 0.5)

    def test_undefined_rates_are_not_invented(self):
        scores = stage7.pixel_metrics(tp=0, fp=0, tn=10, fn=0)
        self.assertIsNone(scores["precision"])
        self.assertIsNone(scores["sensitivity"])
        self.assertEqual(scores["specificity"], 1.0)
        self.assertEqual(scores["dice"], 1.0)


if __name__ == "__main__":
    unittest.main()
