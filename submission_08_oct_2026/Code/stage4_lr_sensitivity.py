"""Screen 16 optimizer/LR configurations on saved fold 1, using MSE only.

Default execution performs preflight only. Training requires the explicit --train
flag and must be launched only after the user approves the experiment.
"""

# Completed MSE recovery learning-rate screening: documented runner.
# 
# Four optimizers and four rates form sixteen configurations on saved Fold 1.
# This is MSE-only evidence; selected rates have not been established for MAE,
# Huber, BCE or Dice. Supervisor-directed future selection may be performed for
# each optimizer-loss pair. The present algorithm and recorded results remain
# unchanged; default execution checks setup and training requires --train.


import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback
from uuid import uuid4

import numpy as np

import experiment_config as config


# Store training/validation MSE and PSNR for every fully logged epoch, including failed runs.
HISTORY_FIELDS = (
    "epoch", "loss", "mse", "psnr_metric", "val_loss", "val_mse", "val_psnr_metric"
)
# Keep raw best epoch separate from the epoch of restored weights and last attempted stopping epoch.
RESULT_FIELDS = (
    "run_id", "optimizer", "learning_rate", "fold", "loss", "run_status",
    "best_epoch", "stopping_epoch", "completed_epochs", "restored_weights_epoch",
    "best_validation_mse", "restored_validation_mse", "validation_psnr",
    "training_time_seconds", "initial_weights_sha256", "history_file", "error",
)


class DivergingRun(RuntimeError):
    """Non-finite loss, MSE, or model parameters detected during one run."""


def weights_digest(weights):
    """Hash array shapes, dtypes and bytes to establish exact initialization identity.
    """
    digest = hashlib.sha256()
    for weight in weights:
        array = np.ascontiguousarray(weight)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def json_safe(value):
    """Convert arrays, paths and non-finite numbers into explicit JSON-safe values.
    
    NaN/Infinity are retained as strings rather than silently discarded or
    emitted as non-standard JSON numeric literals.
    """
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def save_json(path, data):
    """Write complete JSON through a temporary sibling, then replace the destination.
    """
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(data), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def save_csv(path, fields, rows):
    """Write named result/history columns and replace the destination atomically.
    """
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_fresh_model(tf):
    # Import this guarded model definition only; never import a pilot runner.
    """Build the shared compact U-Net after clearing state and resetting seed 42.
    
    Only the guarded architecture module is imported; pilot runners are not
    used as helper modules because some execute training at import time.
    """
    from stage2_unet import build_unet

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config.SEED)
    return build_unet(input_shape=config.INPUT_SHAPE)


def preflight(archive_path=None):
    """Read saved development IDs, instantiate optimizers, build ONE model; no fit."""
    splits = config.load_development_splits(archive_path)
    split_report = {
        "archive": str(splits.archive),
        "development_count": len(splits.development),
        "selected_fold": splits.fold,
        "training_count": len(splits.training),
        "validation_count": len(splits.validation),
        "fresh_test_id_range": [config.FINAL_TEST_START, config.FINAL_TEST_STOP - 1],
        "fresh_test_development_disjoint": True,
        "pilot_test_array_loaded": False,
        "test_images_generated": False,
    }
    print("Split checks passed:")
    print(json.dumps(split_report, indent=2))
    tf = config.tensorflow_runtime()
    tf.keras.utils.set_random_seed(config.SEED)
    determinism = "enabled"
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError) as exc:
        determinism = f"unavailable: {type(exc).__name__}: {exc}"
    optimizer_definitions = []
    # Cross Adam, Adagrad, RMSProp and SGD-Nesterov with 1e-4, 1e-3, 1e-2 and 1e-1.
    for name in config.OPTIMIZERS:
        for learning_rate in config.LEARNING_RATES:
            optimizer = config.make_optimizer(name, learning_rate)
            optimizer_definitions.append({
                "optimizer": name,
                "learning_rate": learning_rate,
                "class": type(optimizer).__name__,
                "settings": optimizer.get_config(),
            })
            del optimizer
    model = build_fresh_model(tf)
    if tuple(model.input_shape[1:]) != config.INPUT_SHAPE:
        raise ValueError("U-Net input shape differs from the fixed protocol")
    if tuple(model.output_shape[1:]) != config.INPUT_SHAPE:
        raise ValueError("U-Net output shape differs from the fixed protocol")
    # One independent weight snapshot is shared exactly across all configurations, not merely a common seed.
    initial_weights = [weight.copy() for weight in model.get_weights()]
    if not all(np.all(np.isfinite(weight)) for weight in initial_weights):
        raise ValueError("Initial U-Net weights are non-finite")
    for weight in initial_weights:
        weight.setflags(write=False)
    report = {
        **split_report,
        "tensorflow_version": tf.__version__,
        "optimizer_configurations_instantiated": len(optimizer_definitions),
        "model_parameters": model.count_params(),
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "initial_weights_sha256": weights_digest(initial_weights),
        "deterministic_ops": determinism,
        "training_started": False,
    }
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    print("Optimizer and U-Net checks passed:")
    print(json.dumps(json_safe(report), indent=2))
    return tf, splits, initial_weights, optimizer_definitions, report


def make_clean_image(tf):
    """Construct the float32 phantom target with one grayscale channel.
    
    This helper is used by the training runner, not lightweight preflight;
    no test image is constructed by this function.
    """
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize

    phantom = resize(
        shepp_logan_phantom(),
        (config.IMAGE_SIZE, config.IMAGE_SIZE),
        anti_aliasing=True,
    ).astype(np.float32)
    return tf.constant(phantom[..., np.newaxis], dtype=tf.float32)


def make_screening_dataset(tf, indices, splits, clean_image, training):
    """Generate only the exact saved Fold 1 training or validation membership.
    
    Each ID uses the stateless seed [42, ID]. Training order is reproducibly
    reshuffled; validation order is fixed. Guards exclude both old pilot
    holdout IDs and fresh final-test IDs from the screening pipeline.
    """
    indices = config.require_screening_ids(indices, splits, training)

    def create_pair(index):
        """Generate one deterministic Gaussian-degraded input and the fixed clean target.
        """
        index = tf.cast(index, tf.int32)
        sample_seed = tf.stack((tf.constant(config.SEED, tf.int32), index))
        noise = tf.random.stateless_normal(
            shape=tf.shape(clean_image), seed=sample_seed,
            mean=config.GAUSSIAN_MEAN, stddev=config.GAUSSIAN_STD, dtype=tf.float32,
        )
        noisy = tf.clip_by_value(
            clean_image + noise, config.PIXEL_MIN, config.PIXEL_MAX
        )
        return noisy, clean_image

    dataset = tf.data.Dataset.from_tensor_slices(indices)
    if training:
        dataset = dataset.shuffle(
            len(indices), seed=config.SEED, reshuffle_each_iteration=True
        )
    dataset = dataset.map(create_pair, num_parallel_calls=1)
    dataset = dataset.batch(config.BATCH_SIZE).prefetch(1)
    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.with_options(options)


def make_psnr_metric(tf):
    """Provide a named Keras PSNR metric for normalized validation images.
    """
    def psnr_metric(y_true, y_pred):
        """Average per-image PSNR using a maximum normalized intensity of one.
        
        PSNR is measured in decibels. Averaging image PSNR is generally different
        from computing PSNR from the dataset's average MSE.
        """
        return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=config.PIXEL_MAX))
    return psnr_metric


def make_recorder(tf):
    """Create an in-memory history recorder with non-finite-value failure detection.
    
    No callback writes to disk inside the timed fit interval. The callback
    also tracks the epoch qualifying under min_delta for weight restoration,
    which may differ from the raw minimum validation-MSE epoch.
    """
    class RunRecorder(tf.keras.callbacks.Callback):
        """Track complete/partial progress and signal divergence within one configuration.
        """
        def __init__(self):
            """Initialize independent per-run history, batch progress and restored-epoch state.
            """
            super().__init__()
            self.rows = []
            self.current_epoch = None
            self.last_training_batch = None
            self.last_validation_batch = None
            self.qualified_best_loss = np.inf
            self.restored_weights_epoch = None

        @staticmethod
        def scalar_logs(logs):
            """Convert callback metric scalars into portable Python numeric values.
            """
            return {key: float(value) for key, value in (logs or {}).items()}

        @staticmethod
        def require_finite_error(logs):
            """Reject non-finite loss/MSE while allowing valid perfect-recovery PSNR limits.
            """
            for key in ("loss", "mse", "val_loss", "val_mse"):
                if key in logs and not np.isfinite(logs[key]):
                    raise DivergingRun(f"Non-finite {key}: {logs[key]}")

        def on_epoch_begin(self, epoch, logs=None):
            """Record the one-based last attempted epoch, including an interrupted epoch.
            """
            self.current_epoch = epoch + 1

        def on_train_batch_end(self, batch, logs=None):
            """Capture training progress and abort only this run if loss/MSE diverges.
            """
            values = self.scalar_logs(logs)
            self.last_training_batch = {
                "epoch": self.current_epoch, "batch": batch + 1, "metrics": values
            }
            self.require_finite_error(values)

        def on_test_batch_end(self, batch, logs=None):
            """Capture validation progress and detect non-finite validation errors.
            """
            values = self.scalar_logs(logs)
            self.last_validation_batch = {
                "epoch": self.current_epoch, "batch": batch + 1, "metrics": values
            }
            self.require_finite_error(values)

        def on_epoch_end(self, epoch, logs=None):
            """Store a full epoch and track min_delta-qualified validation improvement.
            
            Finite model weights are required. The raw best epoch is calculated
            separately from the complete history when the run is saved.
            """
            values = self.scalar_logs(logs)
            self.rows.append({"epoch": epoch + 1, **values})
            self.require_finite_error(values)
            if not all(np.all(np.isfinite(w)) for w in self.model.get_weights()):
                raise DivergingRun("Non-finite model weights")
            validation_mse = values.get("val_mse")
            if validation_mse is None:
                raise RuntimeError("Training history is missing val_mse")
            validation_loss = values.get("val_loss")
            if validation_loss is None:
                raise RuntimeError("Training history is missing val_loss")
            # An improvement smaller than min_delta does not replace restored weights, even if it is a new raw minimum.
            if validation_loss < self.qualified_best_loss - config.MIN_DELTA:
                self.qualified_best_loss = validation_loss
                self.restored_weights_epoch = epoch + 1

    return RunRecorder()


def run_configuration(tf, splits, clean_image, initial_weights, name, lr, directory):
    """Fit one optimizer/rate from the identical shared initialization.
    
    Compilation, data creation, restored-model evaluation and file writing
    are outside the perf_counter interval. Exceptions become saved statuses
    so a divergent rate does not discard other runs. History retains actual
    epochs, raw best validation MSE and the min_delta-qualified restored epoch.
    """
    directory.mkdir(exist_ok=False)
    recorder = make_recorder(tf)
    model = None
    optimizer_settings = None
    initial_hash = weights_digest(initial_weights)
    row = dict.fromkeys(RESULT_FIELDS)
    row.update({
        "run_id": directory.name, "optimizer": name, "learning_rate": lr,
        "fold": splits.fold, "loss": config.LR_SCREENING_LOSS,
        "run_status": "running", "stopping_epoch": 0,
        "training_time_seconds": None, "initial_weights_sha256": initial_hash,
        "history_file": str(directory.relative_to(directory.parent) / "history.csv"),
        "error": "",
    })
    save_json(directory / "run.json", row)
    interrupted = False
    error_trace = None
    try:
        model = build_fresh_model(tf)
        # Assign the same initial array values, then verify their shape/dtype/value hash before every fit.
        model.set_weights(initial_weights)
        if weights_digest(model.get_weights()) != initial_hash:
            raise RuntimeError("Model initialization differs from shared initial weights")
        optimizer = config.make_optimizer(name, lr)
        optimizer_settings = optimizer.get_config()
        model.compile(
            optimizer=optimizer, loss=config.make_loss(config.LR_SCREENING_LOSS),
            metrics=[tf.keras.metrics.MeanSquaredError(name="mse"), make_psnr_metric(tf)],
        )
        training = make_screening_dataset(tf, splits.training, splits, clean_image, True)
        validation = make_screening_dataset(tf, splits.validation, splits, clean_image, False)
        # Uniform 50-epoch ceiling, patience 5, min_delta 1e-5 and restore_best_weights=True are empirical rules.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.MIN_DELTA, restore_best_weights=config.RESTORE_BEST_WEIGHTS,
        )
        # No disk writes are performed by callbacks. Only fit is timed.
        # Only model.fit is timed, including its validation and callback work; setup/evaluation/saving are outside.
        started = time.perf_counter()
        try:
            model.fit(
                training, validation_data=validation, epochs=config.MAX_EPOCHS,
                callbacks=[recorder, early_stopping], verbose=1, shuffle=False,
            )
        finally:
            row["training_time_seconds"] = time.perf_counter() - started
        # Measure validation MSE and mean per-image PSNR after EarlyStopping restores its qualified best weights.
        restored = model.evaluate(validation, verbose=0, return_dict=True)
        restored_mse = float(restored["mse"])
        restored_psnr = float(restored["psnr_metric"])
        if (not np.isfinite(restored_mse) or np.isnan(restored_psnr)
                or restored_psnr == -np.inf):
            raise DivergingRun("Non-finite restored-model validation result")
        row.update({
            "restored_validation_mse": restored_mse,
            "validation_psnr": restored_psnr,
            "run_status": "completed",
        })
    # Record this run's divergence; the outer screen continues rather than abandoning completed configurations.
    except DivergingRun as exc:
        row.update(run_status="diverged", error=str(exc))
        error_trace = traceback.format_exc()
    except KeyboardInterrupt:
        row.update(run_status="interrupted", error="Interrupted by user")
        interrupted = True
    except Exception as exc:
        row.update(run_status="failed", error=f"{type(exc).__name__}: {exc}")
        error_trace = traceback.format_exc()
    finally:
        row["stopping_epoch"] = recorder.current_epoch or 0
        row["completed_epochs"] = len(recorder.rows)
        row["restored_weights_epoch"] = recorder.restored_weights_epoch
        # The raw minimum comes from complete finite validation-history rows, irrespective of min_delta qualification.
        finite_rows = [r for r in recorder.rows if np.isfinite(r.get("val_mse", np.nan))]
        if finite_rows:
            best = min(finite_rows, key=lambda item: item["val_mse"])
            row["best_epoch"] = best["epoch"]
            row["best_validation_mse"] = best["val_mse"]
        fields = list(HISTORY_FIELDS)
        for history_row in recorder.rows:
            for key in history_row:
                if key not in fields:
                    fields.append(key)
        save_csv(directory / "history.csv", fields, recorder.rows)
        save_json(directory / "history.json", recorder.rows)
        save_json(directory / "progress.json", {
            "last_attempted_epoch": recorder.current_epoch,
            "last_training_batch": recorder.last_training_batch,
            "last_validation_batch": recorder.last_validation_batch,
        })
        save_json(directory / "run.json", {**row, "optimizer_settings": optimizer_settings})
        if error_trace:
            (directory / "error.txt").write_text(error_trace, encoding="utf-8")
        del model
        tf.keras.backend.clear_session()
        gc.collect()
    return row, interrupted


def run_screening(preflight_results):
    """Run the sixteen MSE configurations and save provenance plus aggregate results.
    
    This function is called only by the explicitly selected training route.
    Each invocation uses a new timestamp/UUID directory separate from pilots.
    Per-optimizer rates minimize raw best validation MSE over completed runs;
    exact ties choose the smaller rate. No test performance selects a rate.
    """
    tf, splits, initial_weights, optimizer_definitions, check_report = preflight_results
    # Unique output directories isolate revised results from all pilot results and prior screening invocations.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = config.LR_RESULTS_ROOT / f"screening_{stamp}_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    clean_image = make_clean_image(tf)
    save_json(output / "protocol.json", {
        **config.protocol_dict(),
        "split_archive": str(splits.archive),
        "development_and_screening_ids_sha256": weights_digest([
            splits.development, splits.training, splits.validation
        ]),
        "initial_weights_sha256": weights_digest(initial_weights),
        "optimizer_definitions": optimizer_definitions,
        "preflight": check_report,
        "lr_selection_rule": "Lowest best validation MSE among completed runs; "
                             "exact ties use the smaller learning rate",
        "best_epoch_definition": "First raw minimum validation MSE in completed history",
        "restored_weights_epoch_definition": "Best val_loss epoch qualifying under min_delta",
        "divergence_definition": "Non-finite loss, MSE, weights, or invalid restored metric",
        "stopping_epoch_definition": "Last attempted epoch (0 if fit never entered an epoch)",
        "completed_epochs_definition": "Number of fully logged epochs; see progress.json for partial epochs",
        "timing_boundary": "perf_counter around model.fit only",
        "source_sha256": {
            name: hashlib.sha256((config.PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in ("experiment_config.py", "stage4_lr_sensitivity.py", "stage2_unet.py")
        },
    })
    save_json(output / "environment.json", {
        "python": sys.version, "python_executable": sys.executable,
        "tensorflow": tf.__version__, "numpy": np.__version__,
        "keras": getattr(tf.keras, "__version__", "unavailable"),
        "platform": platform.platform(), "processor": platform.processor(),
        "machine": platform.machine(),
        "devices": [str(device) for device in tf.config.list_physical_devices()],
    })
    # Archive only development/Fold 1 membership and initialization; no test images or test metrics are included.
    np.savez(output / "screening_splits.npz", development=splits.development,
             training=splits.training, validation=splits.validation)
    # Archive only development/Fold 1 membership and initialization; no test images or test metrics are included.
    np.savez(output / "initial_weights.npz",
             **{f"weight_{i:03d}": value for i, value in enumerate(initial_weights)})
    rows = []
    # Cross Adam, Adagrad, RMSProp and SGD-Nesterov with 1e-4, 1e-3, 1e-2 and 1e-1.
    for name in config.OPTIMIZERS:
        for lr in config.LEARNING_RATES:
            run_id = f"{name.lower()}_lr_{lr:.0e}"
            print(f"\nScreening {name}, learning rate {lr:g}; outputs: {output / run_id}")
            row, interrupted = run_configuration(
                tf, splits, clean_image, initial_weights, name, lr, output / run_id
            )
            rows.append(row)
            save_csv(output / "lr_screening_results.csv", RESULT_FIELDS, rows)
            save_json(output / "lr_screening_results.json", rows)
            print(f"Run status: {row['run_status']}")
            if interrupted:
                print(f"Experiment interrupted. Partial results saved in {output}")
                return 130
    # Choose each optimizer's MSE-screening rate from completed validation runs only; this is not a universal optimum.
    selections = {}
    # Cross Adam, Adagrad, RMSProp and SGD-Nesterov with 1e-4, 1e-3, 1e-2 and 1e-1.
    for name in config.OPTIMIZERS:
        candidates = [r for r in rows if r["optimizer"] == name
                      and r["run_status"] == "completed"
                      and r["best_validation_mse"] is not None]
        best = min(candidates, key=lambda r: (r["best_validation_mse"], r["learning_rate"])) if candidates else None
        selections[name] = {
            "learning_rate": best["learning_rate"] if best else None,
            "best_validation_mse": best["best_validation_mse"] if best else None,
            "selection_status": "selected" if best else "no_completed_run",
        }
    save_json(output / "selected_learning_rates.json", selections)
    print(f"Screening finished. Results: {output}")
    return 0


def main(argv=None):
    """Choose lightweight preflight by default and require --train for the fitting route.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Lightweight preflight only (default)")
    mode.add_argument("--train", action="store_true", help="Start 16 fits only after explicit user approval")
    args = parser.parse_args(argv)
    try:
        checks = preflight()
        # The default and --check-only routes return before any model.fit or result-directory creation.
        if not args.train:
            print("Preflight complete. No training started and no result files created.")
            return 0
        return run_screening(checks)
    except Exception as exc:
        print(f"Setup/preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Resolve this issue before starting the LR experiment.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
