"""ISIC 2018 Task 1 segmentation runner. No training runs on import or by default.

The runner accepts the official image/mask ZIPs directly or extracted folders.
It keeps train/validation and final-test evaluation as separate commands.
The adjacent protocol is a draft: training is disabled until its choices are
reviewed and ``confirmed_for_training`` is set to true.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from typing import Iterable
from zipfile import ZipFile, is_zipfile


IMAGE_RE = re.compile(r"^(ISIC_\d{7})\.jpe?g$", re.IGNORECASE)
MASK_RE = re.compile(r"^(ISIC_\d{7})_segmentation\.png$", re.IGNORECASE)
DEFAULT_PROTOCOL = Path(__file__).with_name("segmentation_protocol_draft.json")
RESULTS_ROOT = Path(__file__).resolve().parent / "results" / "stage7_isic_segmentation"


@dataclass(frozen=True)
class Pair:
    image_id: str
    image_member: str
    mask_member: str


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    for key in ("seed", "image_size", "batch_size", "maximum_epochs",
                "early_stopping_patience", "early_stopping_min_delta",
                "validation_fraction", "test_fraction", "prediction_threshold"):
        if key not in protocol:
            raise ValueError(f"Protocol is missing {key}")
    for key in ("image_size", "batch_size", "maximum_epochs", "early_stopping_patience"):
        if type(protocol[key]) is not int or protocol[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if protocol["image_size"] % 8:
        raise ValueError("image_size must be divisible by 8 for this three-level U-Net")
    if not 0 < protocol["validation_fraction"] < 1 or not 0 < protocol["test_fraction"] < 1:
        raise ValueError("Validation and test fractions must lie between 0 and 1")
    if protocol["validation_fraction"] + protocol["test_fraction"] >= 1:
        raise ValueError("Validation plus test fraction must be below 1")
    if not 0 < protocol["prediction_threshold"] < 1:
        raise ValueError("Prediction threshold must lie between 0 and 1")
    if protocol["early_stopping_min_delta"] < 0:
        raise ValueError("early_stopping_min_delta must be nonnegative")
    if set(protocol.get("optimizers", [])) != {"Adam", "Adagrad", "RMSProp", "SGD-Nesterov"}:
        raise ValueError("Protocol must list the four intended segmentation optimizers")
    if set(protocol.get("losses", [])) != {"BCE", "Dice"}:
        raise ValueError("Protocol must list BCE and Dice losses")
    if protocol.get("sgd_momentum") != 0.9:
        raise ValueError("SGD-Nesterov momentum must equal 0.9 in this implementation")
    return protocol


def source_members(path: Path, pattern: re.Pattern[str]) -> dict[str, str]:
    """Index matching basenames in one directory or ZIP; detect duplicate IDs."""
    if path.is_dir():
        names = (str(p.relative_to(path)) for p in path.rglob("*") if p.is_file())
    elif path.is_file() and is_zipfile(path):
        with ZipFile(path) as archive:
            names = [item.filename for item in archive.infolist() if not item.is_dir()]
    else:
        raise FileNotFoundError(f"Expected an extracted directory or ZIP: {path}")
    found: dict[str, str] = {}
    for member in names:
        match = pattern.fullmatch(Path(member).name)
        if match:
            image_id = match.group(1).upper()
            if image_id in found:
                raise ValueError(f"Duplicate {image_id} in {path}: {found[image_id]}, {member}")
            found[image_id] = member
    return found


def pair_manifest(images: Path, masks: Path) -> list[Pair]:
    image_members = source_members(images, IMAGE_RE)
    mask_members = source_members(masks, MASK_RE)
    if not image_members or not mask_members:
        raise ValueError("No matching ISIC JPEG images or segmentation PNG masks were found")
    missing_masks = sorted(image_members.keys() - mask_members.keys())
    missing_images = sorted(mask_members.keys() - image_members.keys())
    if missing_masks or missing_images:
        raise ValueError(
            f"Image-mask IDs do not match: {len(missing_masks)} without masks, "
            f"{len(missing_images)} without images. Examples: "
            f"{missing_masks[:4]} / {missing_images[:4]}"
        )
    return [Pair(i, image_members[i], mask_members[i]) for i in sorted(image_members)]


class PairReader:
    """Read original bytes without extracting official ZIPs to disk."""

    def __init__(self, images: Path, masks: Path):
        self.images = images
        self.masks = masks
        self.stack = ExitStack()
        self.image_zip: ZipFile | None = None
        self.mask_zip: ZipFile | None = None

    def __enter__(self) -> PairReader:
        if self.images.is_file():
            self.image_zip = self.stack.enter_context(ZipFile(self.images))
        if self.masks.is_file():
            self.mask_zip = self.stack.enter_context(ZipFile(self.masks))
        return self

    def __exit__(self, *args) -> None:
        self.stack.close()

    def read(self, pair: Pair) -> tuple[bytes, bytes]:
        image = (self.image_zip.read(pair.image_member) if self.image_zip
                 else (self.images / pair.image_member).read_bytes())
        mask = (self.mask_zip.read(pair.mask_member) if self.mask_zip
                else (self.masks / pair.mask_member).read_bytes())
        return image, mask


def make_split(ids: Iterable[str], seed: int, validation_fraction: float,
               test_fraction: float) -> dict[str, list[str]]:
    ordered = sorted(ids)
    if len(ordered) != len(set(ordered)) or len(ordered) < 10:
        raise ValueError("At least 10 unique paired IDs are needed for a train/validation/test split")
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_fraction + 0.5))
    n_validation = max(1, int(len(shuffled) * validation_fraction + 0.5))
    if n_test + n_validation >= len(shuffled):
        raise ValueError("Split fractions leave no training samples")
    result = {
        "train": sorted(shuffled[n_test + n_validation:]),
        "validation": sorted(shuffled[n_test:n_test + n_validation]),
        "test": sorted(shuffled[:n_test]),
    }
    if set(result["train"]) & set(result["validation"]) or set(result["train"]) & set(result["test"]) or set(result["validation"]) & set(result["test"]):
        raise AssertionError("Data split overlap")
    if set().union(*map(set, result.values())) != set(ordered):
        raise AssertionError("Data split does not cover every pair")
    return result


def prepare_splits(pairs: list[Pair], protocol: dict, root: Path) -> tuple[dict, Path]:
    ids = [p.image_id for p in pairs]
    settings = {
        "ids_sha256": sha256_json(ids),
        "seed": protocol["seed"],
        "validation_fraction": protocol["validation_fraction"],
        "test_fraction": protocol["test_fraction"],
    }
    path = root / f"split_{sha256_json(settings)[:16]}.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("settings") != settings:
            raise ValueError("Saved split settings do not match the current dataset")
        split = saved["split"]
        if set().union(*map(set, split.values())) != set(ids):
            raise ValueError("Saved split IDs do not match the current dataset")
        return split, path
    split = make_split(ids, settings["seed"], settings["validation_fraction"], settings["test_fraction"])
    save_json(path, {"settings": settings, "split": split})
    return split, path


def tensorflow_runtime():
    try:
        import tensorflow as tf
    except (ImportError, OSError) as exc:
        raise RuntimeError("A working TensorFlow installation is needed for model checks or training") from exc
    return tf


def decode_original(tf, image_bytes, mask_bytes):
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    mask = tf.io.decode_png(mask_bytes, channels=1)
    tf.debugging.assert_equal(tf.shape(image)[:2], tf.shape(mask)[:2],
                              message="Image and mask dimensions differ")
    tf.debugging.assert_equal(
        tf.reduce_all(tf.logical_or(mask == 0, mask == 255)), True,
        message="Official Task 1 mask values must be 0 or 255",
    )
    return image, mask


def decode_pair(tf, image_bytes, mask_bytes, image_size: int,
                image_id, seed: int, augment: bool):
    image, mask = decode_original(tf, image_bytes, mask_bytes)
    image = tf.image.resize(image, (image_size, image_size), method="bilinear", antialias=True)
    mask = tf.image.resize(mask, (image_size, image_size), method="nearest")
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask > 127, tf.float32)
    if augment:
        digits = tf.strings.to_number(tf.strings.substr(image_id, 5, 7), tf.int32)
        flip_lr = tf.random.stateless_uniform((), seed=tf.stack((seed, digits))) < 0.5
        flip_ud = tf.random.stateless_uniform((), seed=tf.stack((seed + 1, digits))) < 0.5
        image = tf.cond(flip_lr, lambda: tf.image.flip_left_right(image), lambda: image)
        mask = tf.cond(flip_lr, lambda: tf.image.flip_left_right(mask), lambda: mask)
        image = tf.cond(flip_ud, lambda: tf.image.flip_up_down(image), lambda: image)
        mask = tf.cond(flip_ud, lambda: tf.image.flip_up_down(mask), lambda: mask)
    image.set_shape((image_size, image_size, 3))
    mask.set_shape((image_size, image_size, 1))
    return image, mask


def dataset(tf, pairs: list[Pair], ids: list[str], images: Path, masks: Path,
            protocol: dict, training: bool):
    by_id = {pair.image_id: pair for pair in pairs}
    chosen = [by_id[i] for i in ids]
    if training:
        random.Random(protocol["seed"]).shuffle(chosen)

    def generator():
        with PairReader(images, masks) as reader:
            for pair in chosen:
                image_bytes, mask_bytes = reader.read(pair)
                yield image_bytes, mask_bytes, pair.image_id.encode("ascii")

    source = tf.data.Dataset.from_generator(
        generator,
        output_signature=(tf.TensorSpec((), tf.string), tf.TensorSpec((), tf.string),
                          tf.TensorSpec((), tf.string)),
    )
    source = source.map(
        lambda im, ma, identity: decode_pair(
            tf, im, ma, protocol["image_size"], identity, protocol["seed"],
            training and protocol["paired_horizontal_vertical_flips"]),
        num_parallel_calls=1, deterministic=True,
    )
    return source.batch(protocol["batch_size"]).prefetch(1)


def overlap_scores(tf, y_true, y_pred, threshold: float):
    truth = tf.cast(y_true > 0.5, tf.float32)
    prediction = tf.cast(y_pred >= threshold, tf.float32)
    axes = (1, 2, 3)
    intersection = tf.reduce_sum(truth * prediction, axis=axes)
    truth_count = tf.reduce_sum(truth, axis=axes)
    predicted_count = tf.reduce_sum(prediction, axis=axes)
    dice_denominator = truth_count + predicted_count
    union = truth_count + predicted_count - intersection
    dice = tf.where(dice_denominator > 0, 2 * intersection / tf.maximum(dice_denominator, 1), 1.0)
    iou = tf.where(union > 0, intersection / tf.maximum(union, 1), 1.0)
    return dice, iou


def dice_loss(tf, y_true, y_pred):
    axes = (1, 2, 3)
    truth = tf.cast(y_true, tf.float32)
    prediction = tf.cast(y_pred, tf.float32)
    numerator = 2 * tf.reduce_sum(truth * prediction, axis=axes) + 1e-6
    denominator = tf.reduce_sum(truth + prediction, axis=axes) + 1e-6
    return 1.0 - tf.reduce_mean(numerator / denominator)


def optimizer(tf, name: str, learning_rate: float):
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if name == "Adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if name == "Adagrad":
        return tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    if name == "RMSProp":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    if name == "SGD-Nesterov":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimizer: {name}")


def compile_model(tf, model, protocol: dict, optimizer_name: str,
                  loss_name: str, learning_rate: float):
    if optimizer_name not in protocol["optimizers"] or loss_name not in protocol["losses"]:
        raise ValueError("Optimizer or loss is not listed in the reviewed protocol")
    selected_loss = (tf.keras.losses.BinaryCrossentropy() if loss_name == "BCE"
                     else lambda true, predicted: dice_loss(tf, true, predicted))
    threshold = protocol["prediction_threshold"]
    model.compile(
        optimizer=optimizer(tf, optimizer_name, learning_rate),
        loss=selected_loss,
        metrics=[
            tf.keras.metrics.MeanMetricWrapper(
                lambda true, predicted: overlap_scores(tf, true, predicted, threshold)[0], name="dice"),
            tf.keras.metrics.MeanMetricWrapper(
                lambda true, predicted: overlap_scores(tf, true, predicted, threshold)[1], name="iou"),
        ],
    )


def build_model(tf, protocol: dict):
    tf.keras.utils.set_random_seed(protocol["seed"])
    from stage2_unet import build_unet
    return build_unet(input_shape=(protocol["image_size"], protocol["image_size"], 3))


def initial_weights_hash(model) -> str:
    digest = hashlib.sha256()
    for weight in model.get_weights():
        digest.update(weight.tobytes(order="C"))
    return digest.hexdigest()


def inspect_pair(tf, pair: Pair, images: Path, masks: Path, protocol: dict) -> dict:
    with PairReader(images, masks) as reader:
        image_bytes, mask_bytes = reader.read(pair)
    original_image, original_mask = decode_original(tf, image_bytes, mask_bytes)
    resized_image, resized_mask = decode_pair(
        tf, image_bytes, mask_bytes, protocol["image_size"],
        tf.constant(pair.image_id), protocol["seed"], False)
    return {
        "id": pair.image_id,
        "original_shape": original_image.shape.as_list(),
        "mask_shape": original_mask.shape.as_list(),
        "model_image_shape": resized_image.shape.as_list(),
        "model_mask_shape": resized_mask.shape.as_list(),
        "foreground_fraction": float(tf.reduce_mean(resized_mask).numpy()),
    }


def check_only(images: Path, masks: Path, protocol: dict,
               allow_small_sample: bool) -> dict:
    pairs = pair_manifest(images, masks)
    expected = protocol["expected_training_pairs"]
    if len(pairs) != expected and not allow_small_sample:
        raise ValueError(
            f"Found {len(pairs)} pairs, expected {expected}. "
            "Use --allow-small-sample only for a check of a limited fixture."
        )
    tf = tensorflow_runtime()
    inspected = [inspect_pair(tf, pair, images, masks, protocol) for pair in pairs[:min(3, len(pairs))]]
    model = build_model(tf, protocol)
    if model.output_shape != (None, protocol["image_size"], protocol["image_size"], 1):
        raise AssertionError(f"Wrong model output shape: {model.output_shape}")
    for name in protocol["optimizers"]:
        optimizer(tf, name, 1e-3)
    with PairReader(images, masks) as reader:
        image_bytes, mask_bytes = reader.read(pairs[0])
    model_image, model_mask = decode_pair(
        tf, image_bytes, mask_bytes, protocol["image_size"],
        tf.constant(pairs[0].image_id), protocol["seed"], False)
    prediction = model(model_image[None, ...], training=False)
    if tuple(prediction.shape) != (1, protocol["image_size"], protocol["image_size"], 1):
        raise AssertionError("Model forward pass has an unexpected output shape")
    loss_checks = {
        "BCE": float(tf.keras.losses.BinaryCrossentropy()(model_mask[None, ...], prediction).numpy()),
        "Dice": float(dice_loss(tf, model_mask[None, ...], prediction).numpy()),
    }
    for name in protocol["losses"]:
        compile_model(tf, model, protocol, "Adam", name, 1e-3)
    result = {
        "status": "passed",
        "training_started": False,
        "pair_count": len(pairs),
        "expected_full_dataset_pairs": expected,
        "sample_only": len(pairs) != expected,
        "sample_checks": inspected,
        "model_input_shape": model.input_shape,
        "model_output_shape": model.output_shape,
        "model_parameters": model.count_params(),
        "optimizers_instantiated": protocol["optimizers"],
        "losses_available": protocol["losses"],
        "loss_values_from_untrained_forward_pass": loss_checks,
        "protocol_confirmed_for_training": protocol["confirmed_for_training"],
    }
    tf.keras.backend.clear_session()
    return result


def train_one(images: Path, masks: Path, protocol: dict, optimizer_name: str,
              loss_name: str, learning_rate: float) -> Path:
    if not protocol["confirmed_for_training"]:
        raise RuntimeError("Protocol is a draft. Confirm the Sunday decisions in JSON before training.")
    pairs = pair_manifest(images, masks)
    if len(pairs) != protocol["expected_training_pairs"]:
        raise ValueError(f"Real training requires all {protocol['expected_training_pairs']} official paired images")
    if optimizer_name not in protocol["optimizers"] or loss_name not in protocol["losses"]:
        raise ValueError("Optimizer or loss is outside the confirmed protocol")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    split, split_file = prepare_splits(pairs, protocol, RESULTS_ROOT)
    run_key = f"{optimizer_name.lower()}_{loss_name.lower()}_{learning_rate:.0e}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RESULTS_ROOT / f"{stamp}_{run_key}"
    run_dir.mkdir()
    record = {
        "status": "started", "started_at": utc_now(),
        "optimizer": optimizer_name, "loss": loss_name,
        "learning_rate": learning_rate,
        "protocol": protocol, "protocol_sha256": sha256_json(protocol),
        "split_file": str(split_file),
        "dataset_ids_sha256": sha256_json([p.image_id for p in pairs]),
        "counts": {name: len(ids) for name, ids in split.items()},
        "note": "No test image or mask was read during training.",
    }
    save_json(run_dir / "run.json", record)
    tf = tensorflow_runtime()
    try:
        model = build_model(tf, protocol)
        record["initial_weights_sha256"] = initial_weights_hash(model)
        compile_model(tf, model, protocol, optimizer_name, loss_name, learning_rate)
        train_data = dataset(tf, pairs, split["train"], images, masks, protocol, True)
        val_data = dataset(tf, pairs, split["validation"], images, masks, protocol, False)
        stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_dice", mode="max",
            patience=protocol["early_stopping_patience"],
            min_delta=protocol["early_stopping_min_delta"],
            restore_best_weights=True,
        )
        logger = tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv"))
        start = time.perf_counter()
        try:
            history = model.fit(
                train_data.repeat(), validation_data=val_data.repeat(),
                steps_per_epoch=math.ceil(len(split["train"]) / protocol["batch_size"]),
                validation_steps=math.ceil(len(split["validation"]) / protocol["batch_size"]),
                epochs=protocol["maximum_epochs"],
                callbacks=[logger, stopping], verbose=2, shuffle=False,
            )
        finally:
            record["fit_seconds"] = time.perf_counter() - start
        restored_validation = model.evaluate(
            val_data.repeat(),
            steps=math.ceil(len(split["validation"]) / protocol["batch_size"]),
            verbose=0, return_dict=True,
        )
        model.save_weights(run_dir / "best.weights.h5")
        values = history.history.get("val_dice", [])
        record.update(
            status="completed",
            completed_at=utc_now(),
            epochs_run=len(history.epoch),
            raw_best_validation_dice_epoch=(1 + max(range(len(values)), key=values.__getitem__) if values else None),
            restored_validation={key: float(value) for key, value in restored_validation.items()},
        )
    except Exception as exc:
        record.update(status="failed", completed_at=utc_now(),
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_json(run_dir / "run.json", record)
        tf.keras.backend.clear_session()
    return run_dir


def final_evaluate(images: Path, masks: Path, run_dir: Path) -> Path:
    """Evaluate a frozen trained run on its held-out IDs, once."""
    run_file = run_dir / "run.json"
    record = json.loads(run_file.read_text(encoding="utf-8"))
    if record.get("status") != "completed":
        raise ValueError("Only a completed training run can be evaluated")
    if (run_dir / "test_summary.json").exists():
        raise FileExistsError("This run already has final-test results; no repeat evaluation")
    protocol = record["protocol"]
    pairs = pair_manifest(images, masks)
    if sha256_json([p.image_id for p in pairs]) != record["dataset_ids_sha256"]:
        raise ValueError("Dataset IDs changed since training")
    split = json.loads(Path(record["split_file"]).read_text(encoding="utf-8"))["split"]
    by_id = {pair.image_id: pair for pair in pairs}
    tf = tensorflow_runtime()
    model = build_model(tf, protocol)
    model.load_weights(run_dir / "best.weights.h5")
    threshold = protocol["prediction_threshold"]
    predictions = run_dir / "predicted_masks"
    predictions.mkdir()
    rows = []
    with PairReader(images, masks) as reader:
        for image_id in split["test"]:
            image_bytes, mask_bytes = reader.read(by_id[image_id])
            original_image, original_mask = decode_original(tf, image_bytes, mask_bytes)
            model_image = tf.image.resize(
                original_image, (protocol["image_size"], protocol["image_size"]),
                method="bilinear", antialias=True)
            model_image = tf.cast(model_image, tf.float32) / 255.0
            probability = model(model_image[None, ...], training=False)[0]
            probability = tf.image.resize(probability, tf.shape(original_mask)[:2], method="bilinear")
            predicted = tf.cast(probability >= threshold, tf.float32)
            truth = tf.cast(original_mask > 127, tf.float32)
            dice, iou = overlap_scores(tf, truth[None, ...], predicted[None, ...], 0.5)
            raw_iou = float(iou[0].numpy())
            rows.append({
                "image_id": image_id,
                "dice": float(dice[0].numpy()),
                "iou": raw_iou,
                "thresholded_jaccard_0_65": raw_iou if raw_iou >= 0.65 else 0.0,
            })
            png = tf.io.encode_png(tf.cast(predicted * 255, tf.uint8)).numpy()
            (predictions / f"{image_id}_segmentation.png").write_bytes(png)
    metrics_file = run_dir / "test_metrics.csv"
    with metrics_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "evaluated_at": utc_now(), "number_of_test_images": len(rows),
        "threshold": threshold, "evaluation_resolution": "original image/mask pixels",
        "mean_dice": sum(row["dice"] for row in rows) / len(rows),
        "mean_iou": sum(row["iou"] for row in rows) / len(rows),
        "mean_thresholded_jaccard_0_65": sum(row["thresholded_jaccard_0_65"] for row in rows) / len(rows),
    }
    save_json(run_dir / "test_summary.json", summary)
    tf.keras.backend.clear_session()
    return metrics_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True,
                        help="Official training-image ZIP or extracted directory")
    parser.add_argument("--masks", type=Path, required=True,
                        help="Official Task 1 ground-truth ZIP or extracted directory")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--final-evaluate", type=Path, metavar="RUN_DIR")
    parser.add_argument("--allow-small-sample", action="store_true",
                        help="Permit a few matched images for check-only; never for training")
    parser.add_argument("--optimizer", choices=("Adam", "Adagrad", "RMSProp", "SGD-Nesterov"))
    parser.add_argument("--loss", choices=("BCE", "Dice"))
    parser.add_argument("--learning-rate", type=float)
    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol)
    if args.check_only:
        report = check_only(args.images, args.masks, protocol, args.allow_small_sample)
        print(json.dumps(report, indent=2))
        return 0
    if args.train:
        if args.allow_small_sample:
            parser.error("--allow-small-sample is for --check-only only")
        if args.optimizer is None or args.loss is None or args.learning_rate is None:
            parser.error("--train requires --optimizer, --loss and --learning-rate")
        print(train_one(args.images, args.masks, protocol, args.optimizer, args.loss, args.learning_rate))
        return 0
    print(final_evaluate(args.images, args.masks, args.final_evaluate))
    return 0


if __name__ == "__main__":
    sys.exit(main())
