"""Fixed revised recovery protocol. Importing this module never trains a model."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
PILOT_SPLIT_ARCHIVE = (
    PROJECT_ROOT / "results" / "stage3_cross_validation" / "data_splits.npz"
)
LR_RESULTS_ROOT = PROJECT_ROOT / "results" / "stage4_lr_sensitivity"

SEED = 42
IMAGE_SIZE = 512
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1)
GAUSSIAN_MEAN = 0.0
GAUSSIAN_STD = 0.1
PIXEL_MIN = 0.0
PIXEL_MAX = 1.0
BATCH_SIZE = 1
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 1e-5
RESTORE_BEST_WEIGHTS = True

OPTIMIZERS = ("Adam", "Adagrad", "RMSProp", "SGD-Nesterov")
LEARNING_RATES = (1e-4, 1e-3, 1e-2, 1e-1)
LR_SCREENING_LOSS = "MSE"
RECOVERY_LOSSES = ("MSE", "MAE", "Huber")
HUBER_DELTA = 0.1
SGD_MOMENTUM = 0.9
SGD_NESTEROV = True

PILOT_SAMPLE_COUNT = 500
DEVELOPMENT_SAMPLES = 425
CV_FOLDS = 5
SCREENING_FOLD = 1
FOLD_TRAIN_SAMPLES = 340
FOLD_VALIDATION_SAMPLES = 85

# These are reserved identifiers only; screening must never generate their images.
FINAL_TEST_START = 500
FINAL_TEST_STOP = 575  # Exclusive upper bound: IDs 500 through 574.
FINAL_TEST_IDS = tuple(range(FINAL_TEST_START, FINAL_TEST_STOP))
FINAL_TEST_SAMPLES = 75


@dataclass(frozen=True)
class DevelopmentSplits:
    development: np.ndarray
    training: np.ndarray
    validation: np.ndarray
    fold: int
    archive: Path


def _validated_ids(values, expected_count, label):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError(f"{label} must be a one-dimensional integer array")
    if len(values) != expected_count or len(np.unique(values)) != expected_count:
        raise ValueError(f"{label} must contain {expected_count} unique IDs")
    if np.any(values < 0) or np.any(values >= PILOT_SAMPLE_COUNT):
        raise ValueError(f"{label} contains IDs outside the saved development domain")
    result = values.astype(np.int32, copy=True)
    result.setflags(write=False)
    return result


def load_development_splits(archive_path=None):
    """Read only cv_indices and fold 1 arrays; never read pilot test_indices."""
    archive_path = Path(archive_path or PILOT_SPLIT_ARCHIVE).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Saved pilot split archive not found: {archive_path}")
    train_key = f"fold_{SCREENING_FOLD}_train"
    validation_key = f"fold_{SCREENING_FOLD}_validation"
    with np.load(archive_path, allow_pickle=False) as archive:
        for key in ("cv_indices", train_key, validation_key):
            if key not in archive.files:
                raise ValueError(f"Split archive is missing {key}")
        development = _validated_ids(
            archive["cv_indices"], DEVELOPMENT_SAMPLES, "Development split"
        )
        training = _validated_ids(
            archive[train_key], FOLD_TRAIN_SAMPLES, "Screening training split"
        )
        validation = _validated_ids(
            archive[validation_key], FOLD_VALIDATION_SAMPLES, "Screening validation split"
        )
    if np.intersect1d(training, validation).size:
        raise ValueError("Screening training and validation IDs overlap")
    if not np.array_equal(
        np.sort(np.concatenate((training, validation))), np.sort(development)
    ):
        raise ValueError("The selected fold does not partition the saved development IDs")
    new_test_ids = np.asarray(FINAL_TEST_IDS, dtype=np.int32)
    if len(new_test_ids) != FINAL_TEST_SAMPLES:
        raise ValueError("Fresh final test IDs must number exactly 75")
    if np.intersect1d(development, new_test_ids).size:
        raise ValueError("Fresh final test IDs overlap development IDs")
    return DevelopmentSplits(
        development, training, validation, SCREENING_FOLD, archive_path
    )


def require_screening_ids(indices, splits, training):
    """Accept only the exact saved fold, including its order, for screening."""
    indices = np.asarray(indices)
    expected = splits.training if training else splits.validation
    if not np.array_equal(indices, expected):
        raise ValueError("Screening must use the exact saved fold IDs and order")
    return indices.astype(np.int32, copy=True)


def tensorflow_runtime():
    try:
        import tensorflow as tf
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "A working TensorFlow environment is required. "
            f"TensorFlow import failed: {type(exc).__name__}: {exc}"
        ) from exc
    return tf


def make_optimizer(name, learning_rate):
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown revised optimizer: {name}")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    tf = tensorflow_runtime()
    if name == "Adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if name == "Adagrad":
        return tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    if name == "RMSProp":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    return tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=SGD_MOMENTUM, nesterov=SGD_NESTEROV
    )


def make_loss(name):
    if name not in RECOVERY_LOSSES:
        raise ValueError(f"Unknown recovery loss: {name}")
    tf = tensorflow_runtime()
    if name == "MSE":
        return tf.keras.losses.MeanSquaredError()
    if name == "MAE":
        return tf.keras.losses.MeanAbsoluteError()
    return tf.keras.losses.Huber(delta=HUBER_DELTA)


def protocol_dict():
    return {
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "input_shape": INPUT_SHAPE,
        "gaussian_mean": GAUSSIAN_MEAN,
        "gaussian_std": GAUSSIAN_STD,
        "clip_range": [PIXEL_MIN, PIXEL_MAX],
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": EARLY_STOPPING_PATIENCE,
        "min_delta": MIN_DELTA,
        "restore_best_weights": RESTORE_BEST_WEIGHTS,
        "optimizers": OPTIMIZERS,
        "learning_rates": LEARNING_RATES,
        "screening_loss": LR_SCREENING_LOSS,
        "recovery_losses": RECOVERY_LOSSES,
        "huber_delta": HUBER_DELTA,
        "sgd_momentum": SGD_MOMENTUM,
        "sgd_nesterov": SGD_NESTEROV,
        "screening_fold": SCREENING_FOLD,
        "development_samples": DEVELOPMENT_SAMPLES,
        "training_samples": FOLD_TRAIN_SAMPLES,
        "validation_samples": FOLD_VALIDATION_SAMPLES,
        "final_test_id_start": FINAL_TEST_START,
        "final_test_id_stop_exclusive": FINAL_TEST_STOP,
        "final_test_samples": FINAL_TEST_SAMPLES,
        "pilot_test_policy": "Retired: never used by revised screening",
        "final_test_policy": "Reserved for final evaluation after all tuning",
    }
