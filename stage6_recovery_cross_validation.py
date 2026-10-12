"""Compare twelve frozen optimizer/loss pairs on all five saved recovery folds.

Default/--check-only validates saved screening artifacts, five folds, sixty
planned fits, optimizer factories and one U-Net. It never creates datasets,
generates noisy images, evaluates a model or calls fit. Training requires
--train; final refitting and both test sets are outside this script's scope.

    python -B stage6_recovery_cross_validation.py --check-only
    python -B stage6_recovery_cross_validation.py --train
    python -B stage6_recovery_cross_validation.py --train --resume results/stage6_recovery_cross_validation/comparison_...
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
from uuid import uuid4
import zipfile

import numpy as np

import experiment_config as config
import stage5_loss_lr_screening as screening
from stage4_lr_sensitivity import (
    DivergingRun, build_fresh_model, json_safe, make_clean_image,
    make_screening_dataset, weights_digest,
)


RESULTS_ROOT = config.PROJECT_ROOT / "results" / "stage6_recovery_cross_validation"
MSE_SCREEN = config.LR_RESULTS_ROOT / "combined_final_lr_screening"
DEFAULT_STAGE5 = (config.PROJECT_ROOT / "results" / "stage5_loss_lr_screening"
                  / "screening_20261010T054644Z_d2ae382d")
METRICS = ("validation_mse", "validation_rmse", "validation_nrmse_percent",
           "validation_psnr", "validation_ssim", "training_time_seconds",
           "best_epoch", "stopping_epoch")
INVALID_RECORD = (ValueError, KeyError, TypeError, OSError, OverflowError,
                  EOFError, zipfile.BadZipFile)
ProtocolError = screening.ProtocolError


@dataclass
class Setup:
    base: screening.CheckedSetup
    folds: dict
    runs: list
    rates: dict
    stage5_directory: Path
    resume_directory: Path | None
    report: dict

    def for_run(self, run):
        """Give reused record validators the exact fold and its own split hash."""
        split = self.folds[run["fold"]]
        protocol = {**self.base.protocol,
                    "split_sha256": weights_digest([split.development, split.training, split.validation])}
        return replace(self.base, splits=split, protocol=protocol)


def source_digest(path):
    """Ignore CRLF/LF checkout differences, while detecting every content edit."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_folds():
    """Read only saved development/fold arrays; never access pilot test_indices."""
    folds = {}
    with np.load(config.PILOT_SPLIT_ARCHIVE, allow_pickle=False) as archive:
        development = config._validated_ids(archive["cv_indices"], 425, "Development")
        for fold in range(1, 6):
            training = config._validated_ids(archive[f"fold_{fold}_train"], 340, f"Fold {fold} train")
            validation = config._validated_ids(archive[f"fold_{fold}_validation"], 85, f"Fold {fold} validation")
            if np.intersect1d(training, validation).size:
                raise ProtocolError(f"Fold {fold} has training/validation overlap")
            if not np.array_equal(np.sort(np.concatenate((training, validation))), np.sort(development)):
                raise ProtocolError(f"Fold {fold} does not partition all 425 development IDs")
            folds[fold] = config.DevelopmentSplits(
                development, training, validation, fold, config.PILOT_SPLIT_ARCHIVE.resolve())
    all_validation = np.concatenate([split.validation for split in folds.values()])
    if not np.array_equal(np.sort(all_validation), np.sort(development)):
        raise ProtocolError("Each development ID must be validation exactly once across five folds")
    if config.FINAL_TEST_IDS != tuple(range(500, 575)) or np.intersect1d(development, config.FINAL_TEST_IDS).size:
        raise ProtocolError("Fresh test IDs must be exactly 500-574, disjoint from development")
    return folds


def loss_definition(objective, name):
    if name != "MSE":
        return screening.loss_definition(objective, name)
    if type(objective).__name__ != "MeanSquaredError":
        raise ProtocolError("MSE constructor differs from the recovery protocol")
    settings = objective.get_config().copy()
    settings.pop("name", None)
    return json_safe({"class": "MeanSquaredError", "settings": settings, "huber_delta": None})


def audit_stage5(directory, tf, folds, initial_weights):
    """Audit all 32 histories/checkpoints and recompute selections without inference."""
    directory = Path(directory).expanduser().resolve()
    root = (config.PROJECT_ROOT / "results" / "stage5_loss_lr_screening").resolve()
    if directory.parent != root or not directory.name.startswith("screening_"):
        raise ProtocolError("Stage 5 evidence must be a saved Stage 5 screening directory")
    stored = screening.read_json(directory / "protocol.json")
    protocol_hash = stored.pop("protocol_sha256")
    if screening.object_digest(stored) != protocol_hash:
        raise ProtocolError("Stage 5 protocol checksum failed")
    definitions, objectives = {}, {}
    for run in screening.planned_runs():
        definitions[run["run_id"]] = screening.optimizer_definition(
            config.make_optimizer(run["optimizer"], run["learning_rate"]),
            run["optimizer"], run["learning_rate"])
        objectives[run["loss"]] = screening.loss_definition(config.make_loss(run["loss"]), run["loss"])
    canonical = screening.load_weights(directory / "initial_weights.npz", initial_weights)
    if weights_digest(canonical) != weights_digest(initial_weights):
        raise ProtocolError("Fresh seeded U-Net differs from the screening initialization")
    with np.load(directory / "screening_splits.npz", allow_pickle=False) as archive:
        for name in ("development", "training", "validation"):
            if not np.array_equal(archive[name], getattr(folds[1], name)):
                raise ProtocolError("Stage 5 saved Fold 1 differs from the pilot archive")
    expected = screening.make_protocol(folds[1], canonical, definitions, objectives,
                                       stored["software_versions"])
    # The authorized restore changed checkout line endings. Accept only exact
    # raw/LF/CRLF variants of the recorded source bytes, never arbitrary edits.
    eol_only = []
    for name, recorded_hash in stored["source_sha256"].items():
        raw = (config.PROJECT_ROOT / name).read_bytes()
        lf = raw.replace(b"\r\n", b"\n")
        variants = {hashlib.sha256(value).hexdigest() for value in (raw, lf, lf.replace(b"\n", b"\r\n"))}
        if recorded_hash not in variants:
            raise ProtocolError(f"Screening source has a content change: {name}")
        if hashlib.sha256(raw).hexdigest() != recorded_hash:
            eol_only.append(name)
    expected["source_sha256"] = stored["source_sha256"]
    if expected != stored:
        raise ProtocolError("Stage 5 saved protocol or constructor definitions failed validation")
    checked = screening.CheckedSetup(tf, folds[1], canonical, definitions, objectives,
                                      stored, protocol_hash, {}, directory, {}, {})
    rows = [screening.validated_completed_row(directory, run, checked) for run in screening.planned_runs()]
    if screening.read_json(directory / "all_runs.json") != rows:
        raise ProtocolError("Stage 5 aggregate differs from its 32 verified per-run records")
    selections, flat = screening.selected_rates(rows)
    if screening.read_json(directory / "selected_learning_rates.json") != selections:
        raise ProtocolError("Stage 5 selected rates differ from recomputed raw-minimum selections")
    with (directory / "selected_learning_rates.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != 8:
        raise ProtocolError("Stage 5 must have eight selected optimizer/loss pairs")
    for actual, expected_row in zip(csv_rows, flat):
        for name, value in expected_row.items():
            if isinstance(value, (int, float)):
                if float(actual[name]) != value:
                    raise ProtocolError("Stage 5 CSV selections disagree with JSON")
            elif actual[name] != str(value):
                raise ProtocolError("Stage 5 CSV selection identity disagrees with JSON")
    summary = screening.read_json(directory / "experiment_summary.json")
    if (summary.get("experiment_status") != "finished" or summary.get("completed_runs") != 32
            or summary.get("run_status_counts") != {"completed": 32}
            or summary.get("optimizer_loss_pairs_with_a_selected_rate") != 8
            or summary.get("protocol_sha256") != protocol_hash
            or summary.get("final_test_untouched") is not True):
        raise ProtocolError("Stage 5 experiment summary does not describe 32 completed runs")
    rates = {name: {loss: selections[name][loss]["learning_rate"] for loss in ("MAE", "Huber")}
             for name in config.OPTIMIZERS}
    return directory, rates, {"verified_stage5_runs": len(rows), "selected_stage5_pairs": len(flat),
                               "stage5_training_seconds": sum(row["training_time_seconds"] for row in rows),
                               "stage5_source_eol_changes_only": eol_only}


def audit_mse(rates, initial_hash):
    """Retain Stage 4's recorded MSE selection; do not refit or rescreen MSE."""
    with (MSE_SCREEN / "lr_screening_results.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    identities = {(row["optimizer"], float(row["learning_rate"])) for row in rows}
    if len(rows) != 16 or identities != {(name, lr) for name in config.OPTIMIZERS for lr in config.LEARNING_RATES}:
        raise ProtocolError("Stage 4 combined results must contain the unique 16-rate grid")
    for row in rows:
        if (row["run_status"] != "completed" or row["loss"] != "MSE" or int(row["fold"]) != 1
                or row["initial_weights_sha256"] != initial_hash
                or not math.isfinite(float(row["best_validation_mse"]))
                or float(row["best_validation_mse"]) < 0):
            raise ProtocolError("Invalid completed MSE screening record")
    saved = screening.read_json(MSE_SCREEN / "selected_learning_rates.json")
    if set(saved) != set(config.OPTIMIZERS):
        raise ProtocolError("MSE selections must name exactly four optimizers")
    for name in config.OPTIMIZERS:
        best = min((row for row in rows if row["optimizer"] == name),
                   key=lambda row: (float(row["best_validation_mse"]), float(row["learning_rate"])))
        if saved[name] != {"learning_rate": float(best["learning_rate"]),
                           "best_validation_mse": float(best["best_validation_mse"]), "run_id": best["run_id"]}:
            raise ProtocolError(f"Recorded MSE selection disagrees with combined evidence: {name}")
        rates[name]["MSE"] = saved[name]["learning_rate"]
    return rates


def planned_runs(rates):
    return [{"run_id": f"{name.lower()}_{loss.lower()}_fold_{fold}",
             "optimizer": name, "loss": loss, "learning_rate": rates[name][loss], "fold": fold}
            for fold in range(1, 6) for name in config.OPTIMIZERS for loss in config.RECOVERY_LOSSES]


def completed_runs(setup, directory):
    completed, reasons = {}, {}
    for run in setup.runs:
        try:
            completed[run["run_id"]] = screening.validated_completed_row(directory, run, setup.for_run(run))
        except INVALID_RECORD as exc:
            reasons[run["run_id"]] = f"{type(exc).__name__}: {exc}"
    return completed, reasons


def preflight(stage5_directory=DEFAULT_STAGE5, resume=None):
    screening.validate_fixed_protocol()
    if config.CV_FOLDS != 5 or config.RECOVERY_LOSSES != ("MSE", "MAE", "Huber"):
        raise ProtocolError("Recovery comparison requires exactly five folds and three losses")
    if RESULTS_ROOT.resolve() != config.PROJECT_ROOT.resolve() / "results" / "stage6_recovery_cross_validation":
        raise ProtocolError("Stage 6 output must not alias screening or pilot results")
    folds = load_folds()
    tf = config.tensorflow_runtime()
    import skimage
    from skimage.metrics import structural_similarity
    del structural_similarity
    tf.config.experimental.enable_op_determinism()
    model = build_fresh_model(tf)
    try:
        if (model.count_params() != screening.EXPECTED_PARAMETERS
                or sum(int(np.prod(value.shape)) for value in model.trainable_weights) != screening.EXPECTED_PARAMETERS
                or tuple(model.input_shape) != (None, 512, 512, 1)
                or tuple(model.output_shape) != (None, 512, 512, 1)):
            raise ProtocolError("U-Net must have 120681 trainable parameters and 512x512x1 I/O")
        initial = [value.copy() for value in model.get_weights()]
        for value in initial:
            if not np.all(np.isfinite(value)):
                raise ProtocolError("Non-finite canonical weights")
            value.setflags(write=False)
    finally:
        del model
        tf.keras.backend.clear_session()
        gc.collect()
    stage5_directory, rates, audit = audit_stage5(stage5_directory, tf, folds, initial)
    rates = audit_mse(rates, weights_digest(initial))
    runs = planned_runs(rates)
    if len(runs) != 60 or len({run["run_id"] for run in runs}) != 60:
        raise ProtocolError("Expected exactly 60 distinct fold/pair runs")
    definitions, objectives = {}, {}
    for run in runs:
        if run["learning_rate"] not in config.LEARNING_RATES:
            raise ProtocolError("Selected rate is outside the approved grid")
        definitions[run["run_id"]] = screening.optimizer_definition(
            config.make_optimizer(run["optimizer"], run["learning_rate"]), run["optimizer"], run["learning_rate"])
        objectives[run["loss"]] = loss_definition(config.make_loss(run["loss"]), run["loss"])
    stopper, recorder = screening.make_early_stopping(tf), screening.make_raw_best_recorder(tf)
    if stopper.restore_best_weights or stopper.patience != 5 or stopper.min_delta != 1e-5:
        raise ProtocolError("Stopping must use patience 5/min_delta 1e-5, with separate raw-best restoration")
    del stopper, recorder
    versions = {"python": platform.python_version(), "tensorflow": tf.__version__,
                "keras": getattr(tf.keras, "__version__", "unavailable"),
                "numpy": np.__version__, "scikit_image": skimage.__version__}
    protocol = json_safe({
        "schema_version": 1, "stage": 6, "seed": 42, "image_size": 512,
        "gaussian_mean": 0.0, "gaussian_std": 0.1, "clip_range": [0.0, 1.0],
        "batch_size": 1, "max_epochs": 50, "patience": 5, "min_delta": 1e-5,
        "early_stopping_restore_best_weights": False,
        "best_model_rule": "First strict raw minimum val_loss; explicitly restore before validation/saving",
        "initial_weights_sha256": weights_digest(initial),
        "initialization_policy": "Same seed-42 fresh canonical weights for all 60 fits; no screened model weights",
        "split_sha256": weights_digest([folds[1].development] + [value for split in folds.values() for value in (split.training, split.validation)]),
        "fold_split_sha256": {fold: weights_digest([split.development, split.training, split.validation]) for fold, split in folds.items()},
        "selected_learning_rates": rates, "planned_runs": runs,
        "optimizer_definitions": definitions, "loss_definitions": objectives,
        "stage5_directory": stage5_directory.relative_to(config.PROJECT_ROOT).as_posix(),
        "screening_evidence_sha256": {str(path.relative_to(config.PROJECT_ROOT).as_posix()): screening.file_digest(path)
                                      for path in (stage5_directory / "protocol.json", stage5_directory / "all_runs.json",
                                                   stage5_directory / "selected_learning_rates.json",
                                                   MSE_SCREEN / "selected_learning_rates.json", MSE_SCREEN / "lr_screening_results.csv")},
        "software_versions": versions,
        "source_hash_policy": "SHA-256 after CRLF-to-LF normalization",
        "source_sha256": {name: source_digest(config.PROJECT_ROOT / name) for name in
                          ("experiment_config.py", "stage2_unet.py", "stage4_lr_sensitivity.py",
                           "stage5_loss_lr_screening.py", "stage6_recovery_cross_validation.py")},
        "metrics": {"mse": "Mean validation pixel MSE", "rmse": "sqrt(mean validation MSE)",
                    "nrmse_percent": "100 * RMSE", "psnr": "Mean per-image PSNR; max_val=1",
                    "ssim": "Mean per-image SSIM; range=1, win=7, uniform/sample covariance"},
        "fold_summary": "Mean and sample SD (ddof=1); mark complete only if all five folds completed",
        "timing_boundary": "time.perf_counter around model.fit only",
        "test_policy": "Both test sets excluded; IDs 500-574 reserved for later final evaluation",
        "resume_rule": "Skip verified completed runs; restart others at epoch 1 with canonical weights/fresh optimizer",
        "interpretation": "Rates were tuned on Fold 1 of this development pool; this is not independent nested CV",
    })
    protocol_hash = screening.object_digest(protocol)
    directory = None
    if resume is not None:
        directory = Path(resume).expanduser().resolve()
        if (not directory.is_dir() or directory.parent != RESULTS_ROOT.resolve()
                or not directory.name.startswith("comparison_")):
            raise ProtocolError("Resume must name an existing Stage 6 comparison directory")
        stored = screening.read_json(directory / "protocol.json")
        stored_hash = stored.pop("protocol_sha256", None)
        if stored != protocol or stored_hash != protocol_hash or screening.object_digest(stored) != stored_hash:
            raise ProtocolError("Resume protocol, rates, evidence, source or software changed")
        if screening.read_json(directory / "selected_learning_rates.json") != rates:
            raise ProtocolError("Stage 6 saved frozen learning rates differ from screening evidence")
        saved_initial = screening.load_weights(directory / "initial_weights.npz", initial)
        if weights_digest(saved_initial) != weights_digest(initial):
            raise ProtocolError("Stage 6 saved canonical weights differ from seed-42 initialization")
        with np.load(directory / "cv_splits.npz", allow_pickle=False) as archive:
            if not np.array_equal(archive["development"], folds[1].development):
                raise ProtocolError("Stage 6 saved development IDs differ")
            for fold, split in folds.items():
                for name in ("training", "validation"):
                    if not np.array_equal(archive[f"fold_{fold}_{name}"], getattr(split, name)):
                        raise ProtocolError("Stage 6 saved fold membership/order differs")
    report = json_safe({
        "recommendation": "READY", "python_executable": sys.executable,
        "software_versions": versions, "development_ids": 425,
        "folds": [{"fold": fold, "training_ids": 340, "validation_ids": 85,
                   "disjoint": True, "union_equals_development": True} for fold in folds],
        "each_development_id_validated_once": True, "fresh_test_ids": [500, 574],
        "fresh_test_count": 75, "test_development_disjoint": True,
        "pilot_test_array_loaded": False, "test_images_generated": False,
        "datasets_created": False, "model_fit_called": False, "model_evaluation_called": False,
        "training_started": False, "deterministic_operations": "enabled",
        "unet_models_built": 1, "unet_parameters": screening.EXPECTED_PARAMETERS,
        "input_shape": [None, 512, 512, 1], "output_shape": [None, 512, 512, 1],
        "initial_weights_sha256": weights_digest(initial),
        "initialization_identical_for_every_run": True,
        "selected_learning_rates": rates, "optimizer_loss_pairs": 12,
        "planned_fits": 60, "optimizer_loss_factories_instantiated": 60,
        "max_epochs": 50, "patience": 5, "min_delta": 1e-5,
        "early_stopping_restore_best_weights": False, "post_fit_raw_best_restoration": True,
        "sgd_momentum": 0.9, "sgd_nesterov": True, "huber_delta": 0.1,
        "results_root": RESULTS_ROOT, "output_separate_from_prior_stages": True,
        "protocol_sha256": protocol_hash, **audit,
    })
    base = screening.CheckedSetup(tf, folds[1], initial, definitions, objectives,
                                   protocol, protocol_hash, report, directory, {}, {})
    setup = Setup(base, folds, runs, rates, stage5_directory, directory, report)
    if directory is not None:
        verified, reasons = completed_runs(setup, directory)
        report.update({"verified_completed_runs_to_skip": len(verified),
                       "runs_to_restart_at_epoch_1": 60 - len(verified), "restart_reasons": reasons})
    print("Stage 6 check-only/preflight report:")
    print(json.dumps(report, indent=2, allow_nan=False))
    return setup


def run_configuration(setup, clean_image, run, directory, restart_reason=""):
    """Fit one fold/pair and durably record a raw-best result, or its failure."""
    checked = setup.for_run(run)
    tf = checked.tf
    run_directory = directory / run["run_id"]
    if run_directory.resolve() != run_directory:
        raise ProtocolError("Run directory must not alias another location")
    run_directory.mkdir(exist_ok=True)
    attempts = run_directory / "attempts"
    if attempts.resolve() != attempts:
        raise ProtocolError("Attempt directory must not alias another location")
    attempts.mkdir(exist_ok=True)
    attempt = attempts / ("attempt_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    attempt.mkdir(exist_ok=False)
    row = screening.expected_row(run, checked)
    row.update({"run_status": "running", "history_file": (attempt.relative_to(directory) / "history.csv").as_posix(),
                "history_json_file": (attempt.relative_to(directory) / "history.json").as_posix()})
    record = {"schema_version": 1, "stage": 6, "attempt": attempt.name,
              "restart_reason": restart_reason, "result": row,
              "optimizer_definition": checked.optimizer_definitions[run["run_id"]],
              "loss_definition": checked.loss_definitions[run["loss"]]}
    screening.save_json(run_directory / "run.json", record)
    recorder = screening.make_raw_best_recorder(tf)
    model, error_trace, fatal, interrupted = None, None, None, False
    try:
        model = build_fresh_model(tf)
        model.set_weights(checked.initial_weights)
        if weights_digest(model.get_weights()) != checked.protocol["initial_weights_sha256"]:
            raise ProtocolError("Run initialization differs from canonical weights")
        optimizer, objective = config.make_optimizer(run["optimizer"], run["learning_rate"]), config.make_loss(run["loss"])
        if screening.optimizer_definition(optimizer, run["optimizer"], run["learning_rate"]) != record["optimizer_definition"]:
            raise ProtocolError("Optimizer changed after preflight")
        if loss_definition(objective, run["loss"]) != record["loss_definition"]:
            raise ProtocolError("Loss changed after preflight")
        model.compile(optimizer=optimizer, loss=objective, metrics=[tf.keras.metrics.MeanSquaredError(name="mse")])
        training = make_screening_dataset(tf, checked.splits.training, checked.splits, clean_image, True)
        validation = make_screening_dataset(tf, checked.splits.validation, checked.splits, clean_image, False)
        stopper = screening.make_early_stopping(tf)
        started = time.perf_counter()
        try:
            model.fit(training, validation_data=validation, epochs=config.MAX_EPOCHS,
                      callbacks=[recorder, stopper], verbose=1, shuffle=False)
        finally:
            row["training_time_seconds"] = time.perf_counter() - started
        if recorder.best_weights is None:
            raise RuntimeError("Fit produced no finite raw-best epoch")
        model.set_weights(recorder.best_weights)
        best_hash = weights_digest(recorder.best_weights)
        if weights_digest(model.get_weights()) != best_hash:
            raise ProtocolError("Raw-best restoration failed its weight checksum")
        row.update(screening.validation_metrics(tf, model, validation, run["loss"], recorder))
        if weights_digest(model.get_weights()) != best_hash:
            raise ProtocolError("Model weights changed during post-fit validation")
        row["run_status"] = "completed"
    except DivergingRun as exc:
        row.update({"run_status": "diverged", "error": str(exc)})
        error_trace = traceback.format_exc()
    except KeyboardInterrupt:
        row.update({"run_status": "interrupted", "error": "User interruption; restart at epoch 1"})
        interrupted, error_trace = True, traceback.format_exc()
    except ProtocolError as exc:
        row.update({"run_status": "protocol_invalid", "error": str(exc)})
        fatal, error_trace = exc, traceback.format_exc()
    except Exception as exc:
        row.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        error_trace = traceback.format_exc()
    finally:
        try:
            row.update({"stopping_epoch": recorder.current_epoch, "completed_epochs": len(recorder.rows),
                        "best_epoch": recorder.best_epoch,
                        "best_validation_loss": recorder.raw_best_validation_loss if recorder.best_epoch else None,
                        "best_validation_mse_at_epoch": recorder.best_logs["val_mse"] if recorder.best_logs else None})
            fields = list(screening.HISTORY_FIELDS) + sorted({key for value in recorder.rows for key in value} - set(screening.HISTORY_FIELDS))
            screening.save_csv(attempt / "history.csv", fields, recorder.rows)
            screening.save_json(attempt / "history.json", recorder.rows)
            row["history_file_sha256"] = screening.file_digest(attempt / "history.csv")
            row["history_json_file_sha256"] = screening.file_digest(attempt / "history.json")
            if recorder.best_weights is not None:
                screening.save_weights(attempt / "best_weights.npz", recorder.best_weights)
                row.update({"best_weights_file": (attempt.relative_to(directory) / "best_weights.npz").as_posix(),
                            "best_weights_file_sha256": screening.file_digest(attempt / "best_weights.npz"),
                            "best_model_weights_sha256": weights_digest(recorder.best_weights)})
                screening.save_json(attempt / "best_weights_metadata.json", {
                    "best_epoch": row["best_epoch"], "best_validation_loss": row["best_validation_loss"],
                    "model_weights_sha256": row["best_model_weights_sha256"],
                    "archive_sha256": row["best_weights_file_sha256"]})
            screening.save_json(attempt / "progress.json", {"last_attempted_epoch": recorder.current_epoch,
                                                             "last_training_batch": recorder.last_training_batch,
                                                             "last_validation_batch": recorder.last_validation_batch})
            if error_trace:
                screening.atomic_write(attempt / "error.txt", lambda stream: stream.write(error_trace.encode("utf-8")))
            record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            screening.save_json(attempt / "run.json", record)
            screening.save_json(run_directory / "run.json", record)
        finally:
            del model
            tf.keras.backend.clear_session()
            gc.collect()
    return row, interrupted, fatal


def fold_summary(rows):
    """Aggregate common metrics with sample SD; never conceal missing folds."""
    summary = []
    for name in config.OPTIMIZERS:
        for loss in config.RECOVERY_LOSSES:
            pair = [row for row in rows if row["optimizer"] == name and row["loss"] == loss]
            completed = [row for row in pair if row["run_status"] == "completed"]
            value = {"optimizer": name, "loss": loss, "learning_rate": pair[0]["learning_rate"],
                     "planned_folds": 5, "completed_folds": len(completed),
                     "all_five_folds_completed": len(completed) == 5,
                     "run_status_counts": json.dumps(dict(Counter(row["run_status"] for row in pair)), sort_keys=True)}
            for field in METRICS:
                values = [float(row[field]) for row in completed]
                value[field + "_mean"] = float(np.mean(values)) if values else None
                # An infinite PSNR has no finite sample SD; retain its mean and
                # report SD as missing rather than inventing a finite number.
                value[field + "_sd"] = float(np.std(values, ddof=1)) if len(values) >= 2 and all(math.isfinite(x) for x in values) else None
            summary.append(value)
    return summary


def save_aggregates(directory, rows_by_id, setup, status):
    rows = [rows_by_id[run["run_id"]] for run in setup.runs]
    screening.save_csv(directory / "all_runs.csv", screening.RESULT_FIELDS, rows)
    screening.save_json(directory / "all_runs.json", rows)
    summary = fold_summary(rows)
    screening.save_csv(directory / "optimizer_loss_summary.csv", list(summary[0]), summary)
    screening.save_json(directory / "optimizer_loss_summary.json", summary)
    screening.save_json(directory / "experiment_summary.json", {
        "stage": 6, "experiment_status": status, "planned_runs": 60,
        "completed_runs": sum(row["run_status"] == "completed" for row in rows),
        "run_status_counts": dict(Counter(row["run_status"] for row in rows)),
        "pairs_with_all_five_folds_completed": sum(row["all_five_folds_completed"] for row in summary),
        "training_seconds_completed_runs": sum(row["training_time_seconds"] for row in rows if row["run_status"] == "completed"),
        "protocol_sha256": setup.base.protocol_sha256, "final_test_untouched": True,
        "updated_at_utc": datetime.now(timezone.utc).isoformat()})


def noisy_baseline(setup, clean_image):
    """Evaluate development-fold noisy inputs only, outside every fit timer."""
    from skimage.metrics import structural_similarity
    tf, rows = setup.base.tf, []
    for fold, split in setup.folds.items():
        mse_values, psnr_values, ssim_values = [], [], []
        for noisy, clean in make_screening_dataset(tf, split.validation, split, clean_image, False):
            mse_values.append(float(tf.reduce_mean(tf.square(noisy - clean)).numpy()))
            psnr_values.extend(float(value) for value in tf.image.psnr(clean, noisy, max_val=1.0).numpy())
            ssim_values.extend(float(structural_similarity(target[..., 0], prediction[..., 0],
                               data_range=1.0, win_size=7, gaussian_weights=False, use_sample_covariance=True))
                               for target, prediction in zip(clean.numpy(), noisy.numpy()))
        if len(mse_values) != 85 or len(psnr_values) != 85 or len(ssim_values) != 85:
            raise ProtocolError("Noisy baseline must use exactly the saved 85 validation IDs")
        mse = float(np.mean(mse_values))
        rows.append({"fold": fold, "validation_samples": 85, "validation_mse": mse,
                     "validation_rmse": math.sqrt(mse), "validation_nrmse_percent": 100 * math.sqrt(mse),
                     "validation_psnr": float(np.mean(psnr_values)), "validation_ssim": float(np.mean(ssim_values))})
    return rows


def run_comparison(setup):
    if setup.resume_directory is None:
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        directory = RESULTS_ROOT / ("comparison_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
        directory.mkdir(exist_ok=False)
        screening.save_json(directory / "protocol.json", {**setup.base.protocol, "protocol_sha256": setup.base.protocol_sha256})
        screening.save_json(directory / "preflight_report.json", setup.report)
        screening.save_weights(directory / "initial_weights.npz", setup.base.initial_weights)
        arrays = {"development": setup.folds[1].development}
        arrays.update({f"fold_{fold}_{name}": getattr(split, name)
                       for fold, split in setup.folds.items() for name in ("training", "validation")})
        screening.atomic_write(directory / "cv_splits.npz", lambda stream: np.savez(stream, **arrays))
        screening.save_json(directory / "selected_learning_rates.json", setup.rates)
        screening.save_json(directory / "environment.json", {
            **setup.base.protocol["software_versions"], "python_executable": sys.executable,
            "platform": platform.platform(), "processor": platform.processor(),
            "devices": [str(value) for value in setup.base.tf.config.list_physical_devices()]})
    else:
        directory = setup.resume_directory
    with screening.experiment_lock(directory):
        verified, reasons = completed_runs(setup, directory)
        rows = {run["run_id"]: verified.get(run["run_id"], screening.expected_row(run, setup.for_run(run)))
                for run in setup.runs}
        save_aggregates(directory, rows, setup, "running")
        pending = [run for run in setup.runs if run["run_id"] not in verified]
        if not pending:
            save_aggregates(directory, rows, setup, "finished")
            print(f"All 60 runs verify complete; no fit occurred: {directory}")
            return 0
        clean_image = make_clean_image(setup.base.tf)
        baseline = noisy_baseline(setup, clean_image)
        screening.save_csv(directory / "noisy_baseline.csv", list(baseline[0]), baseline)
        screening.save_json(directory / "noisy_baseline.json", baseline)
        for run in pending:
            print(f"\n{run['run_id']}: LR={run['learning_rate']}, canonical initialization, epoch 1")
            row, interrupted, fatal = run_configuration(setup, clean_image, run, directory, reasons.get(run["run_id"], ""))
            rows[run["run_id"]] = row
            if row["run_status"] == "completed":
                try:
                    rows[run["run_id"]] = screening.validated_completed_row(directory, run, setup.for_run(run))
                except INVALID_RECORD as exc:
                    row.update({"run_status": "failed", "error": f"Completed-record validation failed: {exc}"})
                    record = screening.read_json(directory / run["run_id"] / "run.json")
                    record["result"] = row
                    screening.save_json(directory / run["run_id"] / "run.json", record)
            save_aggregates(directory, rows, setup, "interrupted" if interrupted else "protocol_invalid" if fatal else "running")
            print(f"Run status: {row['run_status']}")
            if interrupted:
                print(f"Completed records preserved; resume requires --train --resume '{directory}'")
                return 130
            if fatal:
                raise fatal
        save_aggregates(directory, rows, setup, "finished")
    print(f"All 60 fits attempted; inspect status counts and five-fold summaries: {directory}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Preflight only; default; no fitting/evaluation")
    mode.add_argument("--train", action="store_true", help="Run 60 fold/pair fits after explicit training approval")
    parser.add_argument("--stage5-results", type=Path, default=DEFAULT_STAGE5, help="Completed MAE/Huber screening evidence")
    parser.add_argument("--resume", type=Path, help="Existing Stage 6 comparison; fitting still requires --train")
    args = parser.parse_args(argv)
    try:
        setup = preflight(args.stage5_results, args.resume)
        if not args.train:
            print("READY: no fit, model evaluation, datasets, noisy/test images or Stage 6 result files created.")
            return 0
        return run_comparison(setup)
    except KeyboardInterrupt:
        print("Interrupted; completed per-run records are preserved.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Stage 6 stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
