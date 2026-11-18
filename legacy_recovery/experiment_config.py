"""Fixed revised recovery protocol. Importing this module never trains a model."""

# Revised recovery protocol and constructors. Constants document the existing
# MSE screening and the planned MSE, MAE and Huber comparisons; this module does
# not train or evaluate a model. Fresh final-test IDs 500--574 are identifiers
# reserved for final evaluation, not samples available for tuning.


from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Anchor paths to this file so a different working directory cannot redirect results.
PROJECT_ROOT = Path(__file__).resolve().parent
PILOT_SPLIT_ARCHIVE = (
    PROJECT_ROOT / "results" / "stage3_cross_validation" / "data_splits.npz"
)
LR_RESULTS_ROOT = PROJECT_ROOT / "results" / "stage4_lr_sensitivity"

# Shared protocol: 512-pixel grayscale images; N(0, 0.1^2) noise before [0,1] clipping.
SEED = 42
IMAGE_SIZE = 512
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1)
GAUSSIAN_MEAN = 0.0
GAUSSIAN_STD = 0.1
PIXEL_MIN = 0.0
PIXEL_MAX = 1.0
# Uniform empirical stopping controls, not mathematical convergence theorems.
BATCH_SIZE = 1
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 1e-5
RESTORE_BEST_WEIGHTS = True

# The completed MSE screen crosses four optimizers with four rates (16 trials).
OPTIMIZERS = ("Adam", "Adagrad", "RMSProp", "SGD-Nesterov")
LEARNING_RATES = (1e-4, 1e-3, 1e-2, 1e-1)
LR_SCREENING_LOSS = "MSE"
# Future recovery objectives; only MSE is used in the current screening runner.
RECOVERY_LOSSES = ("MSE", "MAE", "Huber")
HUBER_DELTA = 0.1
# Nesterov momentum distinguishes revised SGD from the historical plain-SGD pilot.
SGD_MOMENTUM = 0.9
SGD_NESTEROV = True

# Retain 425 saved development IDs and five 340-training/85-validation folds.
PILOT_SAMPLE_COUNT = 500
DEVELOPMENT_SAMPLES = 425
CV_FOLDS = 5
SCREENING_FOLD = 1
FOLD_TRAIN_SAMPLES = 340
FOLD_VALIDATION_SAMPLES = 85

# These are reserved identifiers only; screening must never generate their images.
# Fresh final-test IDs are checked as identifiers only and excluded from every tuning stage.
FINAL_TEST_START = 500
FINAL_TEST_STOP = 575  # Exclusive upper bound: IDs 500 through 574.
FINAL_TEST_IDS = tuple(range(FINAL_TEST_START, FINAL_TEST_STOP))
FINAL_TEST_SAMPLES = 75


@dataclass(frozen=True)
class DevelopmentSplits:
    """Immutable container for saved development and Fold 1 membership.
    
    The arrays are validated/read-only copies. No pilot or fresh test-image
    array is part of this screening container.
    """
    development: np.ndarray
    training: np.ndarray
    validation: np.ndarray
    fold: int
    archive: Path


def _validated_ids(values, expected_count, label):
    """Require a unique integer vector of the prescribed size in IDs 0--499.
    
    Return an independent int32 array with writes disabled to protect saved
    split membership from accidental in-place edits.
    """
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
    # Read saved membership instead of regenerating splits; never load the pilot test array.
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
    # Validation leakage is prevented by disjoint sets whose union is the full development pool.
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
    """Import TensorFlow only when a runtime operation actually needs it.
    
    Keeping this dependency lazy allows safe configuration/source inspection
    without constructing a model or invoking a pilot runner.
    """
    try:
        import tensorflow as tf
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "A working TensorFlow environment is required. "
            f"TensorFlow import failed: {type(exc).__name__}: {exc}"
        ) from exc
    return tf


def make_optimizer(name, learning_rate):
    """Construct the selected optimizer with the requested positive rate.
    
    Adam uses first/second gradient moments; Adagrad accumulates squared
    gradients; RMSProp uses an exponential second-moment estimate. The SGD
    branch supplies momentum 0.9 and Nesterov explicitly. Other optimizer
    defaults remain unchanged and are recorded by get_config during screening.
    """
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
    """Create an objective independently of the optimizer choice.
    
    MSE penalizes squared residuals, MAE penalizes absolute residuals, and
    Huber transitions from quadratic to linear penalty at delta=0.1. Stage 4
    selects MSE only; the other losses are available for future comparisons.
    """
    if name not in RECOVERY_LOSSES:
        raise ValueError(f"Unknown recovery loss: {name}")
    tf = tensorflow_runtime()
    if name == "MSE":
        return tf.keras.losses.MeanSquaredError()
    if name == "MAE":
        return tf.keras.losses.MeanAbsoluteError()
    return tf.keras.losses.Huber(delta=HUBER_DELTA)


def protocol_dict():
    """Serialize the fixed settings and test-exclusion policies for provenance.
    """
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
