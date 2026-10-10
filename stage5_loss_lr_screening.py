"""Screen MAE and Huber learning rates on saved recovery Fold 1.

The completed MSE screen is preserved: MSE is never compiled or fitted here.
MAE and Huber have different objectives/scales, so select a learning rate
within each optimizer--loss pair, never by comparing the two loss magnitudes.
Default execution and --check-only build/check initialization without training
or image generation. Every training invocation, including resume, needs --train.

Examples (after separate approval to train):
    python -B stage5_loss_lr_screening.py --check-only
    python -B stage5_loss_lr_screening.py --train
    python -B stage5_loss_lr_screening.py --train --resume results/stage5_loss_lr_screening/screening_...
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from uuid import uuid4
import zipfile

import numpy as np

import experiment_config as config
# Stage 4 has a main guard. Import only its validated, side-effect-free helpers;
# neither its preflight nor its MSE training runner is called by this stage.
from stage4_lr_sensitivity import (
    DivergingRun, build_fresh_model, json_safe, make_clean_image,
    make_screening_dataset, weights_digest,
)


SCREENING_LOSSES = ("MAE", "Huber")
RESULTS_ROOT = config.PROJECT_ROOT / "results" / "stage5_loss_lr_screening"
SCHEMA_VERSION = 1
EXPECTED_PARAMETERS = 120681
HISTORY_FIELDS = ("epoch", "loss", "mse", "val_loss", "val_mse")
RESULT_FIELDS = (
    "run_id", "optimizer", "loss", "learning_rate", "fold", "run_status",
    "best_epoch", "stopping_epoch", "completed_epochs", "best_validation_loss",
    "restored_validation_loss", "best_validation_mse_at_epoch", "validation_mse",
    "validation_rmse", "validation_nrmse_percent", "validation_psnr",
    "validation_ssim", "training_time_seconds", "initial_weights_sha256",
    "split_sha256", "protocol_sha256", "optimizer_definition_sha256",
    "loss_definition_sha256", "history_file", "history_json_file",
    "history_file_sha256", "history_json_file_sha256", "best_weights_file",
    "best_weights_file_sha256", "best_model_weights_sha256", "error",
)
SELECTION_FIELDS = (
    "optimizer", "loss", "selection_status", "completed_rates", "run_id",
    "learning_rate", "best_validation_loss", "best_epoch", "validation_mse",
    "validation_rmse", "validation_nrmse_percent", "validation_psnr",
    "validation_ssim", "training_time_seconds", "best_weights_file",
)


class ProtocolError(RuntimeError):
    """Invalid shared protocol/initialization; fitting must not proceed."""


@dataclass
class CheckedSetup:
    tf: object
    splits: config.DevelopmentSplits
    initial_weights: list
    optimizer_definitions: dict
    loss_definitions: dict
    protocol: dict
    protocol_sha256: str
    report: dict
    resume_directory: Path | None
    completed_rows: dict
    restart_reasons: dict


def planned_runs():
    """Return the fixed 4 x 2 x 4 grid, excluding the already screened MSE."""
    return [
        {"run_id": f"{name.lower()}_{loss.lower()}_lr_{lr:.0e}",
         "optimizer": name, "loss": loss, "learning_rate": lr}
        for name in config.OPTIMIZERS
        for loss in SCREENING_LOSSES
        for lr in config.LEARNING_RATES
    ]


def object_digest(value):
    payload = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, writer):
    """Flush a complete sibling file before replacement; retain the old file on failure."""
    path = Path(path)
    temporary = path.with_name(path.name + f".{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_json(path, data):
    payload = (json.dumps(json_safe(data), indent=2, allow_nan=False) + "\n").encode("utf-8")
    atomic_write(path, lambda stream: stream.write(payload))


def save_csv(path, fields, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, lambda stream: stream.write(buffer.getvalue().encode("utf-8")))


def save_weights(path, weights):
    """Save model arrays only, with no partial/final optimizer state to resume."""
    values = {f"weight_{i:03d}": value for i, value in enumerate(weights)}
    atomic_write(path, lambda stream: np.savez_compressed(stream, **values))


def load_weights(path, template):
    with np.load(path, allow_pickle=False) as archive:
        keys = [f"weight_{i:03d}" for i in range(len(template))]
        if set(archive.files) != set(keys):
            raise ValueError("Weight archive has unexpected array keys")
        weights = [archive[key].copy() for key in keys]
    for value, expected in zip(weights, template):
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError("Weight shape/dtype differs from the canonical U-Net")
        if not np.all(np.isfinite(value)):
            raise ValueError("Weight archive contains non-finite values")
        value.setflags(write=False)
    return weights


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_fixed_protocol():
    """Require the prescribed shared protocol; the Stage 5 restoration override is local."""
    expected = {
        "SEED": 42, "IMAGE_SIZE": 512, "INPUT_SHAPE": (512, 512, 1),
        "GAUSSIAN_MEAN": 0.0, "GAUSSIAN_STD": 0.1, "PIXEL_MIN": 0.0,
        "PIXEL_MAX": 1.0, "BATCH_SIZE": 1, "MAX_EPOCHS": 50,
        "EARLY_STOPPING_PATIENCE": 5, "MIN_DELTA": 1e-5, "HUBER_DELTA": 0.1,
        "SGD_MOMENTUM": 0.9, "SGD_NESTEROV": True, "SCREENING_FOLD": 1,
        "DEVELOPMENT_SAMPLES": 425, "FOLD_TRAIN_SAMPLES": 340,
        "FOLD_VALIDATION_SAMPLES": 85,
        "OPTIMIZERS": ("Adam", "Adagrad", "RMSProp", "SGD-Nesterov"),
        "LEARNING_RATES": (1e-4, 1e-3, 1e-2, 1e-1),
        "FINAL_TEST_IDS": tuple(range(500, 575)),
    }
    for name, value in expected.items():
        if getattr(config, name) != value:
            raise ProtocolError(f"experiment_config.{name} differs from the approved protocol")
    if len(planned_runs()) != 32 or "MSE" in SCREENING_LOSSES:
        raise ProtocolError("Stage 5 must contain exactly 32 MAE/Huber runs")
    expected_root = config.PROJECT_ROOT.resolve() / "results" / "stage5_loss_lr_screening"
    if RESULTS_ROOT.resolve() != expected_root:
        raise ProtocolError("Stage 5 output must not alias another stage's directory")


def optimizer_definition(optimizer, name, lr):
    settings = optimizer.get_config().copy()
    settings.pop("name", None)  # Cosmetic Keras auto-naming is not an optimizer change.
    expected_class = {"Adam": "Adam", "Adagrad": "Adagrad",
                      "RMSProp": "RMSprop", "SGD-Nesterov": "SGD"}[name]
    if type(optimizer).__name__ != expected_class:
        raise ProtocolError(f"Unexpected optimizer class for {name}")
    if not math.isclose(float(settings["learning_rate"]), lr, rel_tol=1e-6):
        raise ProtocolError("Optimizer learning rate differs from the requested grid")
    if name == "SGD-Nesterov" and (
        settings.get("momentum") != 0.9 or settings.get("nesterov") is not True
    ):
        raise ProtocolError("SGD must use momentum 0.9 and Nesterov")
    return json_safe({"class": type(optimizer).__name__, "settings": settings})


def loss_definition(loss_object, name):
    settings = loss_object.get_config().copy()
    settings.pop("name", None)
    expected_class = {"MAE": "MeanAbsoluteError", "Huber": "Huber"}[name]
    if type(loss_object).__name__ != expected_class:
        raise ProtocolError(f"Unexpected loss class for {name}")
    if name == "Huber" and settings.get("delta", config.HUBER_DELTA) != 0.1:
        raise ProtocolError("Huber delta must equal 0.1")
    return json_safe({"class": type(loss_object).__name__, "settings": settings,
                      "huber_delta": config.HUBER_DELTA if name == "Huber" else None})


def resume_directory(value):
    directory = Path(value).expanduser().resolve()
    if (not directory.is_dir() or directory.parent != RESULTS_ROOT.resolve()
            or not directory.name.startswith("screening_")):
        raise ProtocolError("Resume must name an existing Stage 5 screening directory")
    return directory


def make_protocol(splits, initial_weights, optimizer_definitions, loss_definitions, versions):
    shared = config.protocol_dict().copy()
    shared.pop("screening_loss")
    shared.pop("restore_best_weights")
    return json_safe({
        **shared, "schema_version": SCHEMA_VERSION, "stage": 5,
        "screening_losses": SCREENING_LOSSES, "planned_runs": 32,
        "split_archive": str(splits.archive.relative_to(config.PROJECT_ROOT).as_posix()),
        "split_sha256": weights_digest([splits.development, splits.training, splits.validation]),
        "initial_weights_sha256": weights_digest(initial_weights),
        "model_parameters": EXPECTED_PARAMETERS,
        "optimizer_definitions": optimizer_definitions, "loss_definitions": loss_definitions,
        "early_stopping": {"monitor": "val_loss", "mode": "min", "patience": 5,
                           "min_delta": 1e-5, "restore_best_weights": False},
        "best_model_rule": "First strict raw minimum val_loss; no min_delta; restore these weights after fit",
        "lr_selection_rule": "Within each optimizer-loss pair: lowest raw best val_loss; exact ties use smaller LR",
        "timing_boundary": "time.perf_counter around model.fit only, including validation and in-memory callbacks",
        "metrics": {"mse": "Restored-model validation MSE", "rmse": "sqrt(MSE)",
                    "nrmse_percent": "100 * RMSE", "psnr": "Mean per-image tf.image.psnr, max_val=1",
                    "ssim": "Mean per-image skimage.metrics.structural_similarity, data_range=1, win_size=7, uniform window, sample covariance"},
        "resume_rule": "Skip only verified completed runs; restart every other run from epoch 1 and canonical model weights",
        "deterministic_ops": "enabled", "software_versions": versions,
        "source_sha256": {name: file_digest(config.PROJECT_ROOT / name) for name in
                          ("experiment_config.py", "stage2_unet.py", "stage4_lr_sensitivity.py",
                           "stage5_loss_lr_screening.py")},
    })


def preflight(resume=None):
    """Build one canonical U-Net; check 32 factories and saved IDs; never fit or create images."""
    validate_fixed_protocol()
    splits = config.load_development_splits()  # Reads only development/Fold 1 arrays.
    tf = config.tensorflow_runtime()
    import skimage
    from skimage.metrics import structural_similarity  # Verify the metric dependency only.

    del structural_similarity
    tf.keras.utils.set_random_seed(config.SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError) as exc:
        raise ProtocolError(f"Deterministic TensorFlow operations cannot be enabled: {exc}") from exc
    optimizer_definitions, loss_definitions = {}, {}
    for run in planned_runs():
        optimizer = config.make_optimizer(run["optimizer"], run["learning_rate"])
        objective = config.make_loss(run["loss"])
        optimizer_definitions[run["run_id"]] = optimizer_definition(
            optimizer, run["optimizer"], run["learning_rate"]
        )
        definition = loss_definition(objective, run["loss"])
        if run["loss"] in loss_definitions and loss_definitions[run["loss"]] != definition:
            raise ProtocolError("The loss constructor is inconsistent across configurations")
        loss_definitions[run["loss"]] = definition
        del optimizer, objective
    stopper = make_early_stopping(tf)
    recorder = make_raw_best_recorder(tf)
    if stopper.restore_best_weights or stopper.patience != 5 or stopper.min_delta != 1e-5:
        raise ProtocolError("EarlyStopping must only control stopping, without restoring weights")
    del stopper, recorder
    model = build_fresh_model(tf)
    try:
        if tuple(model.input_shape[1:]) != config.INPUT_SHAPE or tuple(model.output_shape[1:]) != config.INPUT_SHAPE:
            raise ProtocolError("U-Net input/output shape differs from 512 x 512 x 1")
        trainable = sum(int(np.prod(weight.shape)) for weight in model.trainable_weights)
        if model.count_params() != EXPECTED_PARAMETERS or trainable != EXPECTED_PARAMETERS:
            raise ProtocolError("U-Net must contain 120681 trainable parameters")
        initial_weights = [value.copy() for value in model.get_weights()]
        for value in initial_weights:
            if not np.all(np.isfinite(value)):
                raise ProtocolError("Canonical initial weights are non-finite")
            value.setflags(write=False)
        input_shape, output_shape = model.input_shape, model.output_shape
    finally:
        del model
        tf.keras.backend.clear_session()
        gc.collect()
    directory = resume_directory(resume) if resume is not None else None
    stored = None
    if directory is not None:
        stored = read_json(directory / "protocol.json")
        initial_weights = load_weights(directory / "initial_weights.npz", initial_weights)
        if weights_digest(initial_weights) != stored.get("initial_weights_sha256"):
            raise ProtocolError("Saved canonical initialization does not match the experiment")
        with np.load(directory / "screening_splits.npz", allow_pickle=False) as archive:
            for name in ("development", "training", "validation"):
                if not np.array_equal(archive[name], getattr(splits, name)):
                    raise ProtocolError("Saved Stage 5 split differs from the fixed development fold")
    versions = {"python": platform.python_version(), "tensorflow": tf.__version__,
                "keras": getattr(tf.keras, "__version__", "unavailable"),
                "numpy": np.__version__, "scikit_image": skimage.__version__}
    protocol = make_protocol(splits, initial_weights, optimizer_definitions, loss_definitions, versions)
    protocol_hash = object_digest(protocol)
    if stored is not None:
        stored_hash = stored.pop("protocol_sha256", None)
        if object_digest(stored) != stored_hash or stored != protocol or stored_hash != protocol_hash:
            raise ProtocolError("Resume protocol/source/software differs from the recorded experiment")
    report = json_safe({
        "development_samples": len(splits.development), "selected_screening_fold": splits.fold,
        "training_samples": len(splits.training), "validation_samples": len(splits.validation),
        "training_validation_disjoint": True, "fold_union_equals_development": True,
        "final_fresh_test_range": [500, 574], "final_fresh_test_samples": 75,
        "final_test_development_disjoint": True, "final_test_untouched": True,
        "pilot_test_array_loaded": False, "test_images_generated": False,
        "screening_images_generated": False, "optimizers": 4, "losses": 2,
        "screening_losses": SCREENING_LOSSES, "learning_rates": config.LEARNING_RATES,
        "learning_rate_count": 4, "total_screening_runs": 32,
        "optimizer_loss_lr_configurations_instantiated": 32, "huber_delta": 0.1,
        "raw_best_and_early_stopping_callbacks_instantiated": True,
        "sgd_momentum": 0.9, "sgd_nesterov": True,
        "max_epochs": 50, "patience": 5, "min_delta": 1e-5,
        "early_stopping_restore_best_weights": False, "post_fit_restore_raw_best_weights": True,
        "tensorflow_version": tf.__version__, "software_versions": versions,
        "python_executable": sys.executable, "deterministic_operations": "enabled",
        "unet_parameter_count": EXPECTED_PARAMETERS, "trainable_parameter_count": trainable,
        "input_shape": input_shape, "output_shape": output_shape,
        "initial_weights_sha256": weights_digest(initial_weights),
        "split_sha256": protocol["split_sha256"], "protocol_sha256": protocol_hash,
        "canonical_weights_source": "saved Stage 5 experiment" if directory else "fresh seeded U-Net",
        "results_root": RESULTS_ROOT, "results_separate_from_stage3_and_stage4": True,
        "mse_will_be_rerun": False, "training_started": False,
    })
    setup = CheckedSetup(tf, splits, initial_weights, optimizer_definitions, loss_definitions,
                         protocol, protocol_hash, report, directory, {}, {})
    if directory is not None:
        setup.completed_rows, setup.restart_reasons = inspect_completed_runs(directory, setup)
        report.update({"resume_directory": str(directory),
                       "verified_completed_runs_to_skip": len(setup.completed_rows),
                       "runs_to_start_from_epoch_1": 32 - len(setup.completed_rows),
                       "restart_reasons": setup.restart_reasons})
    print("Stage 5 preflight report:")
    print(json.dumps(report, indent=2, allow_nan=False))
    return setup


def make_raw_best_recorder(tf):
    """Track every strict raw improvement independently of EarlyStopping.min_delta."""
    class RawBestRecorder(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.rows = []
            self.current_epoch = 0
            self.last_training_batch = None
            self.last_validation_batch = None
            self.raw_best_validation_loss = math.inf
            self.best_epoch = None
            self.best_weights = None
            self.best_logs = None

        @staticmethod
        def finite_logs(logs):
            values = {key: float(value) for key, value in (logs or {}).items()}
            for key in ("loss", "mse", "val_loss", "val_mse"):
                if key in values and not math.isfinite(values[key]):
                    raise DivergingRun(f"Non-finite {key}: {values[key]}")
            return values

        def finite_weights(self):
            values = self.model.get_weights()
            if not all(np.all(np.isfinite(value)) for value in values):
                raise DivergingRun("Non-finite model weights")
            return values

        def on_epoch_begin(self, epoch, logs=None):
            self.current_epoch = epoch + 1
            self.last_training_batch = None
            self.last_validation_batch = None

        def on_train_batch_end(self, batch, logs=None):
            self.last_training_batch = batch + 1
            self.finite_logs(logs)
            self.finite_weights()

        def on_test_batch_end(self, batch, logs=None):
            self.last_validation_batch = batch + 1
            self.finite_logs(logs)

        def on_epoch_end(self, epoch, logs=None):
            values = {key: float(value) for key, value in (logs or {}).items()}
            self.rows.append({"epoch": epoch + 1, **values})
            self.finite_logs(values)
            if not all(key in values for key in HISTORY_FIELDS[1:]):
                raise RuntimeError("Epoch history must contain loss, mse, val_loss and val_mse")
            weights = self.finite_weights()
            # Even an improvement smaller than min_delta replaces this snapshot.
            if values["val_loss"] < self.raw_best_validation_loss:
                self.raw_best_validation_loss = values["val_loss"]
                self.best_epoch = epoch + 1
                self.best_logs = values.copy()
                self.best_weights = [value.copy() for value in weights]
                for value in self.best_weights:
                    value.setflags(write=False)

    return RawBestRecorder()


def make_early_stopping(tf):
    """Use min_delta only for stopping; the raw-best callback owns restoration."""
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.MIN_DELTA, restore_best_weights=False,
    )


def expected_row(run, setup):
    row = dict.fromkeys(RESULT_FIELDS)
    row.update({
        **run, "fold": setup.splits.fold, "run_status": "pending",
        "stopping_epoch": 0, "completed_epochs": 0, "error": "",
        "initial_weights_sha256": setup.protocol["initial_weights_sha256"],
        "split_sha256": setup.protocol["split_sha256"],
        "protocol_sha256": setup.protocol_sha256,
        "optimizer_definition_sha256": object_digest(setup.optimizer_definitions[run["run_id"]]),
        "loss_definition_sha256": object_digest(setup.loss_definitions[run["loss"]]),
    })
    return row


def artifact_path(directory, run_id, relative):
    """Accept only this run's saved attempt files; reject paths into any other stage."""
    if not isinstance(relative, str):
        raise ValueError("Missing artifact path")
    parts = Path(relative).parts
    if (len(parts) != 4 or parts[0] != run_id or parts[1] != "attempts"
            or not parts[2].startswith("attempt_") or ".." in parts):
        raise ValueError("Artifact must belong to the selected run's attempt directory")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("Missing artifact or artifact path escapes the experiment")
    return path


def validated_completed_row(directory, run, setup):
    """Check metadata and saved arrays/history without inference or evaluation."""
    record_path = directory / run["run_id"] / "run.json"
    record = read_json(record_path)
    if not isinstance(record, dict) or not isinstance(record.get("result"), dict):
        raise ValueError("Run record is incomplete or malformed")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported run record schema")
    row = record["result"]
    if row.get("run_status") != "completed":
        raise ValueError(f"Previous status is {row.get('run_status', 'incomplete')}")
    expected = expected_row(run, setup)
    for field in (
        "run_id", "optimizer", "loss", "learning_rate", "fold",
        "initial_weights_sha256", "split_sha256", "protocol_sha256",
        "optimizer_definition_sha256", "loss_definition_sha256",
    ):
        if row.get(field) != expected[field]:
            raise ValueError(f"Run identity/protocol mismatch: {field}")
    if record.get("optimizer_definition") != setup.optimizer_definitions[run["run_id"]]:
        raise ValueError("Saved optimizer settings differ from the checked factory")
    if record.get("loss_definition") != setup.loss_definitions[run["loss"]]:
        raise ValueError("Saved loss definition differs from the checked factory")
    artifacts = {}
    for field in ("history_file", "history_json_file", "best_weights_file"):
        path = artifact_path(directory, run["run_id"], row.get(field))
        if file_digest(path) != row.get(field + "_sha256"):
            raise ValueError(f"Artifact checksum mismatch: {field}")
        artifacts[field] = path
    attempts = {path.parent for path in artifacts.values()}
    if len(attempts) != 1:
        raise ValueError("Artifacts come from different run attempts")
    history = read_json(artifacts["history_json_file"])
    if not isinstance(history, list) or not 1 <= len(history) <= config.MAX_EPOCHS:
        raise ValueError("Completed history is empty or has an invalid epoch count")
    for epoch, values in enumerate(history, start=1):
        if not isinstance(values, dict):
            raise ValueError("Malformed epoch history entry")
        if values.get("epoch") != epoch:
            raise ValueError("History does not start at epoch 1 or skips an epoch")
        for field in HISTORY_FIELDS[1:]:
            if not math.isfinite(float(values[field])) or float(values[field]) < 0:
                raise ValueError("Completed history contains non-finite loss/MSE")
    best = min(history, key=lambda values: values["val_loss"])
    if (row.get("completed_epochs") != len(history)
            or row.get("stopping_epoch") != len(history)
            or row.get("best_epoch") != best["epoch"]
            or row.get("best_validation_loss") != best["val_loss"]
            or row.get("best_validation_mse_at_epoch") != best["val_mse"]):
        raise ValueError("Saved epoch/loss does not identify the first raw history minimum")
    with artifacts["history_file"].open(newline="", encoding="utf-8") as stream:
        csv_history = list(csv.DictReader(stream))
    if len(csv_history) != len(history):
        raise ValueError("CSV and JSON history lengths differ")
    for csv_epoch, json_epoch in zip(csv_history, history):
        if int(csv_epoch["epoch"]) != json_epoch["epoch"]:
            raise ValueError("CSV and JSON epoch numbering differs")
        for field in HISTORY_FIELDS[1:]:
            if float(csv_epoch[field]) != json_epoch[field]:
                raise ValueError("CSV and JSON history metrics differ")
    best_weights = load_weights(artifacts["best_weights_file"], setup.initial_weights)
    if weights_digest(best_weights) != row.get("best_model_weights_sha256"):
        raise ValueError("Saved best model weights fail their numeric checksum")
    checkpoint = read_json(artifacts["best_weights_file"].parent / "best_weights_metadata.json")
    if checkpoint != {
        "best_epoch": row["best_epoch"], "best_validation_loss": row["best_validation_loss"],
        "model_weights_sha256": row["best_model_weights_sha256"],
        "archive_sha256": row["best_weights_file_sha256"],
    }:
        raise ValueError("Checkpoint metadata does not match the raw-best epoch/weights")
    for field in ("best_validation_loss", "restored_validation_loss", "validation_mse",
                  "validation_rmse", "validation_nrmse_percent", "validation_ssim",
                  "training_time_seconds"):
        if not math.isfinite(float(row[field])):
            raise ValueError(f"Completed metric/timing is non-finite: {field}")
    if row["validation_mse"] < 0 or row["training_time_seconds"] < 0:
        raise ValueError("Negative MSE or training time")
    if not math.isclose(row["restored_validation_loss"], row["best_validation_loss"],
                        rel_tol=1e-5, abs_tol=1e-8):
        raise ValueError("Restored validation loss differs from the raw-best loss")
    if not math.isclose(row["validation_mse"], row["best_validation_mse_at_epoch"],
                        rel_tol=1e-5, abs_tol=1e-8):
        raise ValueError("Restored validation MSE differs from the raw-best epoch's MSE")
    if not math.isclose(row["validation_rmse"], math.sqrt(row["validation_mse"]), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("RMSE is inconsistent with MSE")
    if not math.isclose(row["validation_nrmse_percent"], 100 * row["validation_rmse"], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("NRMSE percentage is inconsistent with RMSE")
    psnr = float(row["validation_psnr"])
    if math.isnan(psnr) or psnr == -math.inf or not -1.000001 <= row["validation_ssim"] <= 1.000001:
        raise ValueError("Invalid PSNR or SSIM")
    return {field: row.get(field) for field in RESULT_FIELDS}


def inspect_completed_runs(directory, setup):
    completed, restart_reasons = {}, {}
    for run in planned_runs():
        try:
            completed[run["run_id"]] = validated_completed_row(directory, run, setup)
        except (ValueError, KeyError, TypeError, OSError, OverflowError, EOFError, zipfile.BadZipFile) as exc:
            restart_reasons[run["run_id"]] = f"{type(exc).__name__}: {exc}"
    return completed, restart_reasons


def validation_metrics(tf, model, validation, objective_name, recorder):
    """Evaluate only Fold 1, after exact raw-best restoration; outside the fit timer."""
    from skimage.metrics import structural_similarity

    restored = model.evaluate(validation, verbose=0, return_dict=True)
    restored_loss, mse = float(restored["loss"]), float(restored["mse"])
    if not math.isfinite(restored_loss) or not math.isfinite(mse):
        raise DivergingRun("Non-finite restored-model validation loss/MSE")
    if not math.isclose(restored_loss, recorder.raw_best_validation_loss, rel_tol=1e-5, abs_tol=1e-8):
        raise RuntimeError(f"Restored {objective_name} loss does not match the raw-best epoch")
    if not math.isclose(mse, recorder.best_logs["val_mse"], rel_tol=1e-5, abs_tol=1e-8):
        raise RuntimeError("Restored MSE does not match the raw-best epoch")
    psnr_values, ssim_values = [], []
    for noisy, targets in validation:
        predictions = model(noisy, training=False)
        if not bool(tf.reduce_all(tf.math.is_finite(predictions)).numpy()):
            raise DivergingRun("Non-finite validation predictions")
        psnr_values.extend(float(value) for value in tf.image.psnr(targets, predictions, max_val=1.0).numpy())
        for target, prediction in zip(targets.numpy(), predictions.numpy()):
            ssim_values.append(float(structural_similarity(
                target[..., 0], prediction[..., 0], data_range=1.0,
                win_size=7, gaussian_weights=False, use_sample_covariance=True,
            )))
    if len(psnr_values) != config.FOLD_VALIDATION_SAMPLES or len(ssim_values) != config.FOLD_VALIDATION_SAMPLES:
        raise ProtocolError("Validation dataset does not contain exactly 85 images")
    psnr, ssim = float(np.mean(psnr_values)), float(np.mean(ssim_values))
    # Positive infinite PSNR is the mathematically valid limit for zero image MSE.
    if math.isnan(psnr) or psnr == -math.inf or not math.isfinite(ssim):
        raise DivergingRun("Invalid restored validation PSNR/SSIM")
    if mse < 0:
        raise DivergingRun("Validation MSE must be non-negative")
    rmse = math.sqrt(mse)
    return {"restored_validation_loss": restored_loss, "validation_mse": mse,
            "validation_rmse": rmse, "validation_nrmse_percent": 100.0 * rmse,
            "validation_psnr": psnr, "validation_ssim": ssim}


def run_configuration(setup, clean_image, run, directory, restart_reason=""):
    """Fit one fresh run; preserve old attempts and serialize only outside model.fit."""
    tf = setup.tf
    run_directory = directory / run["run_id"]
    if run_directory.resolve() != run_directory:
        raise ProtocolError("A run directory must not alias another directory")
    run_directory.mkdir(exist_ok=True)
    attempts = run_directory / "attempts"
    if attempts.resolve() != attempts:
        raise ProtocolError("Attempt storage must not alias another directory")
    attempts.mkdir(exist_ok=True)
    attempt_name = "attempt_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    attempt = attempts / attempt_name
    attempt.mkdir(exist_ok=False)
    row = expected_row(run, setup)
    row.update({"run_status": "running",
                "history_file": (attempt.relative_to(directory) / "history.csv").as_posix(),
                "history_json_file": (attempt.relative_to(directory) / "history.json").as_posix()})
    record = {"schema_version": SCHEMA_VERSION, "attempt": attempt_name,
              "restart_reason": restart_reason, "result": row,
              "optimizer_definition": setup.optimizer_definitions[run["run_id"]],
              "loss_definition": setup.loss_definitions[run["loss"]]}
    save_json(run_directory / "run.json", record)
    recorder = make_raw_best_recorder(tf)
    model, error_trace, fatal, interrupted = None, None, None, False
    try:
        model = build_fresh_model(tf)
        model.set_weights(setup.initial_weights)
        if weights_digest(model.get_weights()) != setup.protocol["initial_weights_sha256"]:
            raise ProtocolError("Fresh model differs from canonical initial weights")
        optimizer = config.make_optimizer(run["optimizer"], run["learning_rate"])
        objective = config.make_loss(run["loss"])
        if optimizer_definition(optimizer, run["optimizer"], run["learning_rate"]) != record["optimizer_definition"]:
            raise ProtocolError("Optimizer settings changed after preflight")
        if loss_definition(objective, run["loss"]) != record["loss_definition"]:
            raise ProtocolError("Loss settings changed after preflight")
        model.compile(optimizer=optimizer, loss=objective,
                      metrics=[tf.keras.metrics.MeanSquaredError(name="mse")])
        training = make_screening_dataset(tf, setup.splits.training, setup.splits, clean_image, True)
        validation = make_screening_dataset(tf, setup.splits.validation, setup.splits, clean_image, False)
        early_stopping = make_early_stopping(tf)
        # No callback does disk I/O. This interval contains only the fit call;
        # validation and callback work inside fit are naturally included.
        started = time.perf_counter()
        try:
            model.fit(training, validation_data=validation, epochs=config.MAX_EPOCHS,
                      callbacks=[recorder, early_stopping], verbose=1, shuffle=False)
        finally:
            row["training_time_seconds"] = time.perf_counter() - started
        if recorder.best_weights is None:
            raise RuntimeError("Fit ended without a finite raw-best validation epoch")
        model.set_weights(recorder.best_weights)
        best_hash = weights_digest(recorder.best_weights)
        if weights_digest(model.get_weights()) != best_hash:
            raise ProtocolError("Raw-best restoration failed its exact weight checksum")
        row.update(validation_metrics(tf, model, validation, run["loss"], recorder))
        if weights_digest(model.get_weights()) != best_hash:
            raise ProtocolError("Model weights changed during post-fit validation")
        row["run_status"] = "completed"
    except DivergingRun as exc:
        row.update({"run_status": "diverged", "error": str(exc)})
        error_trace = traceback.format_exc()
    except KeyboardInterrupt:
        row.update({"run_status": "interrupted", "error": "User interruption; restart from epoch 1"})
        interrupted = True
        error_trace = traceback.format_exc()
    except ProtocolError as exc:
        row.update({"run_status": "protocol_invalid", "error": str(exc)})
        fatal = exc
        error_trace = traceback.format_exc()
    except Exception as exc:
        row.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        error_trace = traceback.format_exc()
    finally:
        try:
            row.update({"stopping_epoch": recorder.current_epoch, "completed_epochs": len(recorder.rows),
                        "best_epoch": recorder.best_epoch,
                        "best_validation_loss": recorder.raw_best_validation_loss if recorder.best_epoch is not None else None,
                        "best_validation_mse_at_epoch": recorder.best_logs["val_mse"] if recorder.best_logs else None})
            fields = list(HISTORY_FIELDS) + sorted({key for values in recorder.rows for key in values} - set(HISTORY_FIELDS))
            save_csv(attempt / "history.csv", fields, recorder.rows)
            save_json(attempt / "history.json", recorder.rows)
            row["history_file_sha256"] = file_digest(attempt / "history.csv")
            row["history_json_file_sha256"] = file_digest(attempt / "history.json")
            if recorder.best_weights is not None:
                save_weights(attempt / "best_weights.npz", recorder.best_weights)
                row.update({"best_weights_file": (attempt.relative_to(directory) / "best_weights.npz").as_posix(),
                            "best_weights_file_sha256": file_digest(attempt / "best_weights.npz"),
                            "best_model_weights_sha256": weights_digest(recorder.best_weights)})
                save_json(attempt / "best_weights_metadata.json", {
                    "best_epoch": row["best_epoch"], "best_validation_loss": row["best_validation_loss"],
                    "model_weights_sha256": row["best_model_weights_sha256"],
                    "archive_sha256": row["best_weights_file_sha256"],
                })
            save_json(attempt / "progress.json", {"last_attempted_epoch": recorder.current_epoch,
                                                   "last_training_batch": recorder.last_training_batch,
                                                   "last_validation_batch": recorder.last_validation_batch})
            if error_trace:
                atomic_write(attempt / "error.txt", lambda stream: stream.write(error_trace.encode("utf-8")))
            record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            save_json(attempt / "run.json", record)
            # Publish completed status last, after every required artifact is durable.
            save_json(run_directory / "run.json", record)
        finally:
            del model
            tf.keras.backend.clear_session()
            gc.collect()
    return row, interrupted, fatal


def selected_rates(rows):
    """Select by loss only within each pair; exact ties choose the smaller rate."""
    selections, flat = {}, []
    for name in config.OPTIMIZERS:
        selections[name] = {}
        for loss in SCREENING_LOSSES:
            candidates = [row for row in rows if row["optimizer"] == name and row["loss"] == loss
                          and row["run_status"] == "completed"]
            best = min(candidates, key=lambda row: (row["best_validation_loss"], row["learning_rate"])) if candidates else None
            value = {field: best.get(field) if best else None for field in SELECTION_FIELDS}
            value.update({"optimizer": name, "loss": loss,
                          "selection_status": "selected" if best else "no_completed_run",
                          "completed_rates": len(candidates)})
            selections[name][loss] = value
            flat.append(value)
    return selections, flat


def save_aggregates(directory, rows_by_id, setup, status):
    rows = [rows_by_id[run["run_id"]] for run in planned_runs()]
    save_csv(directory / "all_runs.csv", RESULT_FIELDS, rows)
    save_json(directory / "all_runs.json", rows)
    selections, flat = selected_rates(rows)
    save_json(directory / "selected_learning_rates.json", selections)
    save_csv(directory / "selected_learning_rates.csv", SELECTION_FIELDS, flat)
    save_json(directory / "experiment_summary.json", {
        "stage": 5, "experiment_status": status, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "planned_runs": 32, "run_status_counts": dict(Counter(row["run_status"] for row in rows)),
        "completed_runs": sum(row["run_status"] == "completed" for row in rows),
        "optimizer_loss_pairs_with_a_selected_rate": sum(row["selection_status"] == "selected" for row in flat),
        "training_started": any(row["training_time_seconds"] is not None for row in rows),
        "initial_weights_sha256": setup.protocol["initial_weights_sha256"],
        "split_sha256": setup.protocol["split_sha256"], "protocol_sha256": setup.protocol_sha256,
        "screening_losses": SCREENING_LOSSES, "mse_rerun": False, "final_test_untouched": True,
        "selection_rule": setup.protocol["lr_selection_rule"],
        "selection_scope": "Completed rates within each optimizer-loss pair; no cross-loss ranking",
        "completed_run_training_seconds": sum(row["training_time_seconds"] for row in rows if row["run_status"] == "completed"),
    })


@contextmanager
def experiment_lock(directory):
    """OS advisory lock prevents two training processes from modifying one experiment."""
    with (directory / ".training.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ProtocolError("This Stage 5 experiment is already locked by another process") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        # Keep the lock inode: deleting it could let another process lock a
        # different inode while an existing process still holds this one.


def run_screening(setup):
    """Called exclusively by --train; check-only never reaches this function."""
    if setup.resume_directory is None:
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        directory = RESULTS_ROOT / ("screening_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
        directory.mkdir(exist_ok=False)
        save_json(directory / "protocol.json", {**setup.protocol, "protocol_sha256": setup.protocol_sha256})
        save_json(directory / "preflight_report.json", setup.report)
        save_weights(directory / "initial_weights.npz", setup.initial_weights)
        atomic_write(directory / "screening_splits.npz", lambda stream: np.savez(
            stream, development=setup.splits.development,
            training=setup.splits.training, validation=setup.splits.validation,
        ))
        save_json(directory / "environment.json", {
            **setup.protocol["software_versions"], "python_executable": sys.executable,
            "platform": platform.platform(), "processor": platform.processor(),
            "devices": [str(value) for value in setup.tf.config.list_physical_devices()],
        })
    else:
        directory = setup.resume_directory
    with experiment_lock(directory):
        # Per-run durable records, not a possibly interrupted aggregate, are
        # authoritative. Recheck after acquiring the lock before skipping.
        completed, restart_reasons = inspect_completed_runs(directory, setup)
        rows = {run["run_id"]: completed.get(run["run_id"], expected_row(run, setup)) for run in planned_runs()}
        for run_id, reason in restart_reasons.items():
            rows[run_id]["error"] = "Restart from epoch 1 required: " + reason
        save_aggregates(directory, rows, setup, "running")
        pending = [run for run in planned_runs() if run["run_id"] not in completed]
        if not pending:
            save_aggregates(directory, rows, setup, "finished")
            print(f"All 32 runs are already verified complete. No fit occurred: {directory}")
            return 0
        clean_image = make_clean_image(setup.tf)  # No noisy/test samples are generated here.
        for run in pending:
            print(f"\n{run['run_id']}: start at epoch 1 from canonical weights; outputs: {directory}")
            row, interrupted, fatal = run_configuration(
                setup, clean_image, run, directory, restart_reasons.get(run["run_id"], "")
            )
            rows[run["run_id"]] = row
            if row["run_status"] == "completed":
                # Audit every newly finished record before allowing selection.
                try:
                    rows[run["run_id"]] = validated_completed_row(directory, run, setup)
                except (ValueError, KeyError, TypeError, OSError, OverflowError, EOFError, zipfile.BadZipFile) as exc:
                    row.update({"run_status": "failed", "error": f"Completed-record validation failed: {exc}"})
                    record = read_json(directory / run["run_id"] / "run.json")
                    record["result"] = row
                    save_json(directory / run["run_id"] / "run.json", record)
                    rows[run["run_id"]] = row
            status = "interrupted" if interrupted else ("protocol_invalid" if fatal else "running")
            save_aggregates(directory, rows, setup, status)
            print(f"Run status: {row['run_status']}")
            if interrupted:
                print(f"Completed work is preserved. Resume requires --train --resume '{directory}'")
                return 130
            if fatal is not None:
                raise fatal
        save_aggregates(directory, rows, setup, "finished")
        print(f"All 32 configurations have been attempted. Check status counts and selections: {directory}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Lightweight checks only (default)")
    mode.add_argument("--train", action="store_true", help="Fit the 32 MAE/Huber configurations after explicit approval")
    parser.add_argument("--resume", type=Path, help="Existing Stage 5 experiment; fitting still requires --train")
    args = parser.parse_args(argv)
    try:
        setup = preflight(args.resume)
        if not args.train:
            print("Preflight complete: training_started=false; no fit, inference, noise generation or result files.")
            return 0
        return run_screening(setup)
    except KeyboardInterrupt:
        print("Interrupted. Completed per-run records remain available for verified resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Stage 5 stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Resolve this issue before training. Completed Stage 3/4 files are untouched.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
