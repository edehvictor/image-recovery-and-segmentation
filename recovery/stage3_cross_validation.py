"""Historical five-fold MSE recovery pilot and final pilot refits.

Sample IDs specify repeatable noise realizations of one clean phantom, not
different anatomical images. The 425 development IDs are partitioned into
five 340/85 folds; the original 75 pilot test IDs were evaluated here and are
retired from the revised experiment. This pilot uses three optimizers, plain
SGD, at most 10 epochs and patience 2. Revised settings belong to Stage 4;
none of this pilot's methods or saved numerical results is changed. Execution
at module level includes training, evaluation and writing existing outputs.
"""

import os
import csv
import gc
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import KFold
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

from stage2_unet import build_unet


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

SEED = 42

IMAGE_SIZE = 512
NUM_SAMPLES = 500

TEST_SAMPLES = 75
CV_SAMPLES = NUM_SAMPLES - TEST_SAMPLES

N_FOLDS = 5

BATCH_SIZE = 1

# Pilot stopping controls are intentionally retained; Stage 4's 50/5 controls are different.
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 2

GAUSSIAN_MEAN = 0.0
GAUSSIAN_STD = 0.1


RESULTS_DIR = "results/stage3_cross_validation"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


# ============================================================
# CLEAN SHEPP-LOGAN PHANTOM
# ============================================================

phantom = shepp_logan_phantom()

phantom = resize(
    phantom,
    (IMAGE_SIZE, IMAGE_SIZE),
    anti_aliasing=True
)

phantom = phantom.astype(np.float32)

# Add grayscale channel
phantom = phantom[..., np.newaxis]

clean_image = tf.constant(
    phantom,
    dtype=tf.float32
)


print("Clean phantom shape:", phantom.shape)
print(
    "Pixel range:",
    phantom.min(),
    "to",
    phantom.max()
)


# ============================================================
# GENERATE NOISY/CLEAN PAIR
# ============================================================

def create_noisy_pair(index):

    """Map a sample ID to its repeatable noisy input and shared clean target.
    
    The stateless seed [42, sample_id] fixes that ID's Gaussian realization.
    Noise has mean zero and standard deviation 0.1 before clipping; clipping
    changes the effective error distribution near intensity limits.
    """
    index = tf.cast(index, tf.int32)

    sample_seed = tf.stack([
        tf.constant(SEED, dtype=tf.int32),
        index
    ])

    noise = tf.random.stateless_normal(
        shape=tf.shape(clean_image),
        seed=sample_seed,
        mean=GAUSSIAN_MEAN,
        stddev=GAUSSIAN_STD,
        dtype=tf.float32
    )

    noisy_image = clean_image + noise

    noisy_image = tf.clip_by_value(
        noisy_image,
        0.0,
        1.0
    )

    return noisy_image, clean_image


# ============================================================
# DATASET CREATION
# ============================================================

def make_dataset(indices, training=False):

    """Create a memory-controlled pipeline of paired images from integer IDs.
    
    Only training traversal is shuffled. Pair generation is deterministic per
    ID, batches contain one sample, and prefetching is limited to one batch.
    Membership is supplied by the caller rather than selected from metrics.
    """
    dataset = tf.data.Dataset.from_tensor_slices(
        np.asarray(indices, dtype=np.int32)
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(indices),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        create_noisy_pair,
        num_parallel_calls=1
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(1)

    return dataset


# ============================================================
# PSNR METRIC
# ============================================================

def psnr_metric(y_true, y_pred):

    """Average per-image PSNR using a maximum normalized intensity of one.
    
    PSNR is measured in decibels. Averaging image PSNR is generally different
    from computing PSNR from the dataset's average MSE.
    """
    psnr = tf.image.psnr(
        y_true,
        y_pred,
        max_val=1.0
    )

    return tf.reduce_mean(psnr)


# ============================================================
# OPTIMIZERS
# ============================================================

def get_optimizer(name):

    """Instantiate the original pilot optimizer at its original learning rate.
    
    SGD is plain SGD at 0.01; RMSProp and Adam use 0.001. No Adagrad or
    Nesterov setting is introduced into this historical pilot.
    """
    if name == "SGD":

        return tf.keras.optimizers.SGD(
            learning_rate=0.01
        )

    elif name == "RMSProp":

        return tf.keras.optimizers.RMSprop(
            learning_rate=0.001
        )

    elif name == "Adam":

        return tf.keras.optimizers.Adam(
            learning_rate=0.001
        )

    else:

        raise ValueError(
            "Unknown optimizer: " + name
        )


OPTIMIZERS = [
    "SGD",
    "RMSProp",
    "Adam"
]


# ============================================================
# CREATE FIXED TEST SET AND CV POOL
# ============================================================

all_indices = np.arange(
    NUM_SAMPLES,
    dtype=np.int32
)

rng = np.random.default_rng(SEED)

shuffled_indices = rng.permutation(
    all_indices
)


# The first 75 shuffled IDs form the exposed pilot holdout, retired from revised final testing.
test_indices = shuffled_indices[
    :TEST_SAMPLES
]

# The remaining 425 IDs are the saved development pool reused by revised screening.
cv_indices = shuffled_indices[
    TEST_SAMPLES:
]


print()
print("Total samples:", NUM_SAMPLES)
print("Cross-validation samples:", len(cv_indices))
print("Final test samples:", len(test_indices))


# ============================================================
# CREATE THE FIVE FOLDS ONCE
# ============================================================

# Construct the five folds once; all optimizers reuse identical fold membership.
kfold = KFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED
)


folds = []

# KFold returns positions in the development array; map them back to sample IDs.
for train_positions, val_positions in kfold.split(cv_indices):

    fold_train_indices = cv_indices[
        train_positions
    ]

    fold_val_indices = cv_indices[
        val_positions
    ]

    folds.append(
        (
            fold_train_indices,
            fold_val_indices
        )
    )


print()
print("Cross-validation structure:")

for fold_number, (train_idx, val_idx) in enumerate(
    folds,
    start=1
):

    print(
        f"Fold {fold_number}: "
        f"{len(train_idx)} training, "
        f"{len(val_idx)} validation"
    )


# ============================================================
# SAVE DATA SPLITS
# ============================================================

# Persist membership for reproducibility; Stage 4 reads only development and selected fold keys.
split_data = {
    "test_indices": test_indices,
    "cv_indices": cv_indices
}

for i, (train_idx, val_idx) in enumerate(
    folds,
    start=1
):

    split_data[f"fold_{i}_train"] = train_idx
    split_data[f"fold_{i}_validation"] = val_idx


np.savez(
    os.path.join(
        RESULTS_DIR,
        "data_splits.npz"
    ),
    **split_data
)


# ============================================================
# CALCULATE NOISY TEST-SET BASELINE
# ============================================================

# This baseline and later pilot evaluations expose the old holdout; they are historical evidence.
test_dataset = make_dataset(
    test_indices,
    training=False
)


baseline_mse_values = []
baseline_psnr_values = []


for noisy_batch, clean_batch in test_dataset:

    mse_value = tf.reduce_mean(
        tf.square(
            clean_batch - noisy_batch
        )
    )

    psnr_value = tf.reduce_mean(
        tf.image.psnr(
            clean_batch,
            noisy_batch,
            max_val=1.0
        )
    )

    baseline_mse_values.append(
        float(mse_value.numpy())
    )

    baseline_psnr_values.append(
        float(psnr_value.numpy())
    )


baseline_mse = np.mean(
    baseline_mse_values
)

baseline_psnr = np.mean(
    baseline_psnr_values
)


print()
print("========================================")
print("NOISY INPUT BASELINE")
print("========================================")

print(
    f"Baseline MSE: "
    f"{baseline_mse:.6f}"
)

print(
    f"Baseline PSNR: "
    f"{baseline_psnr:.4f} dB"
)


# ============================================================
# STORAGE FOR RESULTS
# ============================================================

all_fold_results = []

optimizer_summaries = []


# ============================================================
# CROSS-VALIDATION
# ============================================================

# Compare original pilot optimizers under MSE; no optimizer-loss factorial study is completed here.
for optimizer_name in OPTIMIZERS:

    print()
    print("=" * 70)
    print(
        f"STARTING OPTIMIZER: "
        f"{optimizer_name}"
    )
    print("=" * 70)


    optimizer_fold_mse = []
    optimizer_fold_psnr = []
    optimizer_best_epochs = []


    for fold_number, (
        train_indices,
        val_indices
    ) in enumerate(
        folds,
        start=1
    ):

        print()
        print("-" * 60)
        print(
            f"{optimizer_name} - "
            f"FOLD {fold_number}/{N_FOLDS}"
        )
        print("-" * 60)

        print(
            "Training samples:",
            len(train_indices)
        )

        print(
            "Validation samples:",
            len(val_indices)
        )


        # ----------------------------------------------------
        # CLEAR PREVIOUS MODEL
        # ----------------------------------------------------

        tf.keras.backend.clear_session()
        gc.collect()


        # ----------------------------------------------------
        # SAME INITIALIZATION FOR EACH OPTIMIZER
        # ON THE SAME FOLD
        # ----------------------------------------------------

        # Reset the same fold-specific seed for each optimizer; preserve this historical initialization policy.
        fold_seed = SEED + fold_number

        tf.keras.utils.set_random_seed(
            fold_seed
        )


        # ----------------------------------------------------
        # DATASETS
        # ----------------------------------------------------

        train_dataset = make_dataset(
            train_indices,
            training=True
        )

        val_dataset = make_dataset(
            val_indices,
            training=False
        )


        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = build_unet(
            input_shape=(
                IMAGE_SIZE,
                IMAGE_SIZE,
                1
            )
        )


        # ----------------------------------------------------
        # COMPILE
        # ----------------------------------------------------

        model.compile(

            optimizer=get_optimizer(
                optimizer_name
            ),

            loss="mse",

            metrics=[
                tf.keras.metrics.MeanSquaredError(
                    name="mse"
                ),

                psnr_metric
            ]
        )


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        # Validation controls stopping and restored weights; test performance is not the stopping criterion.
        early_stopping = (
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                min_delta=1e-5
            )
        )


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        start_time = time.time()


        history = model.fit(

            train_dataset,

            validation_data=val_dataset,

            epochs=MAX_EPOCHS,

            callbacks=[
                early_stopping
            ],

            verbose=1,

            # Dataset itself is already shuffled.
            shuffle=False
        )


        training_time = (
            time.time() - start_time
        )


        # ----------------------------------------------------
        # BEST EPOCH
        # ----------------------------------------------------

        validation_losses = (
            history.history["val_loss"]
        )

        best_epoch = (
            int(
                np.argmin(
                    validation_losses
                )
            )
            + 1
        )


        optimizer_best_epochs.append(
            best_epoch
        )


        # ----------------------------------------------------
        # EVALUATE VALIDATION SET
        # ----------------------------------------------------

        validation_results = model.evaluate(
            val_dataset,
            verbose=0,
            return_dict=True
        )


        val_mse = float(
            validation_results["mse"]
        )

        val_psnr = float(
            validation_results["psnr_metric"]
        )


        optimizer_fold_mse.append(
            val_mse
        )

        optimizer_fold_psnr.append(
            val_psnr
        )


        print()
        print(
            f"{optimizer_name} "
            f"Fold {fold_number} Results"
        )

        print(
            f"Best epoch: "
            f"{best_epoch}"
        )

        print(
            f"Validation MSE: "
            f"{val_mse:.6f}"
        )

        print(
            f"Validation PSNR: "
            f"{val_psnr:.4f} dB"
        )

        print(
            f"Training time: "
            f"{training_time:.2f} seconds"
        )


        # ----------------------------------------------------
        # STORE FOLD RESULT
        # ----------------------------------------------------

        all_fold_results.append({

            "optimizer":
                optimizer_name,

            "fold":
                fold_number,

            "best_epoch":
                best_epoch,

            "validation_mse":
                val_mse,

            "validation_psnr":
                val_psnr,

            "training_time_seconds":
                training_time
        })


        # ----------------------------------------------------
        # SAVE HISTORY FOR THIS FOLD
        # ----------------------------------------------------

        history_path = os.path.join(

            RESULTS_DIR,

            f"{optimizer_name.lower()}_"
            f"fold_{fold_number}_history.csv"
        )


        with open(
            history_path,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "epoch",
                "training_loss",
                "training_mse",
                "training_psnr",
                "validation_loss",
                "validation_mse",
                "validation_psnr"
            ])


            number_of_epochs = len(
                history.history["loss"]
            )


            for epoch in range(
                number_of_epochs
            ):

                writer.writerow([

                    epoch + 1,

                    history.history[
                        "loss"
                    ][epoch],

                    history.history[
                        "mse"
                    ][epoch],

                    history.history[
                        "psnr_metric"
                    ][epoch],

                    history.history[
                        "val_loss"
                    ][epoch],

                    history.history[
                        "val_mse"
                    ][epoch],

                    history.history[
                        "val_psnr_metric"
                    ][epoch]
                ])


        del model
        gc.collect()


    # ========================================================
    # CROSS-VALIDATION SUMMARY FOR OPTIMIZER
    # ========================================================

    cv_mse_mean = np.mean(
        optimizer_fold_mse
    )

    cv_mse_std = np.std(
        optimizer_fold_mse,
        ddof=1
    )

    cv_psnr_mean = np.mean(
        optimizer_fold_psnr
    )

    cv_psnr_std = np.std(
        optimizer_fold_psnr,
        ddof=1
    )


    # Historical final refit uses rounded mean best epochs; do not relabel it as the future median rule.
    recommended_epochs = max(

        1,

        int(
            round(
                np.mean(
                    optimizer_best_epochs
                )
            )
        )
    )


    print()
    print("=" * 60)

    print(
        f"{optimizer_name} "
        f"CROSS-VALIDATION SUMMARY"
    )

    print("=" * 60)

    print(
        f"Mean validation MSE: "
        f"{cv_mse_mean:.6f} "
        f"± {cv_mse_std:.6f}"
    )

    print(
        f"Mean validation PSNR: "
        f"{cv_psnr_mean:.4f} "
        f"± {cv_psnr_std:.4f} dB"
    )

    print(
        f"Recommended final epochs: "
        f"{recommended_epochs}"
    )


    # ========================================================
    # FINAL TRAINING ON ALL 425 CV SAMPLES
    # ========================================================

    print()
    print(
        f"Training final {optimizer_name} "
        f"model on all {CV_SAMPLES} "
        f"cross-validation samples..."
    )


    tf.keras.backend.clear_session()
    gc.collect()


    tf.keras.utils.set_random_seed(
        SEED + 1000
    )


    final_train_dataset = make_dataset(
        cv_indices,
        training=True
    )


    final_model = build_unet(

        input_shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            1
        )
    )


    final_model.compile(

        optimizer=get_optimizer(
            optimizer_name
        ),

        loss="mse",

        metrics=[
            tf.keras.metrics.MeanSquaredError(
                name="mse"
            ),

            psnr_metric
        ]
    )


    # Refit on all 425 development IDs using pilot-selected epochs; never rerun while preparing submission.
    final_model.fit(

        final_train_dataset,

        epochs=recommended_epochs,

        verbose=1,

        shuffle=False
    )


    # ========================================================
    # FINAL TEST
    # ========================================================

    # Report this as exposed-pilot test performance, not performance on fresh IDs 500--574.
    test_results = final_model.evaluate(
        test_dataset,
        verbose=0,
        return_dict=True
    )


    test_mse = float(
        test_results["mse"]
    )

    test_psnr = float(
        test_results["psnr_metric"]
    )


    print()
    print("=" * 60)

    print(
        f"{optimizer_name} "
        f"FINAL TEST RESULTS"
    )

    print("=" * 60)

    print(
        f"Test MSE: "
        f"{test_mse:.6f}"
    )

    print(
        f"Test PSNR: "
        f"{test_psnr:.4f} dB"
    )


    # ========================================================
    # SAVE FINAL MODEL WEIGHTS
    # ========================================================

    # Existing optimizer-specific checkpoint names would be overwritten by rerunning the pilot.
    final_model.save_weights(

        os.path.join(

            RESULTS_DIR,

            f"{optimizer_name.lower()}"
            f"_final.weights.h5"
        )
    )


    # ========================================================
    # SAVE ONE RECOVERY EXAMPLE
    # ========================================================

    # All pilot panels use the first old-test sample under the same deterministic image generator.
    for noisy_batch, clean_batch in test_dataset.take(1):

        predicted_batch = final_model.predict(
            noisy_batch,
            verbose=0
        )

        noisy_example = (
            noisy_batch[0]
            .numpy()
            .squeeze()
        )

        clean_example = (
            clean_batch[0]
            .numpy()
            .squeeze()
        )

        recovered_example = (
            predicted_batch[0]
            .squeeze()
        )


    plt.figure(
        figsize=(12, 4)
    )


    plt.subplot(1, 3, 1)

    plt.imshow(
        clean_example,
        cmap="gray"
    )

    plt.title(
        "Clean Phantom"
    )

    plt.axis("off")


    plt.subplot(1, 3, 2)

    plt.imshow(
        noisy_example,
        cmap="gray"
    )

    plt.title(
        "Noisy Input"
    )

    plt.axis("off")


    plt.subplot(1, 3, 3)

    plt.imshow(
        recovered_example,
        cmap="gray"
    )

    plt.title(
        f"Recovered - {optimizer_name}"
    )

    plt.axis("off")


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            RESULTS_DIR,

            f"{optimizer_name.lower()}"
            f"_recovery_example.png"
        ),

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()


    # ========================================================
    # STORE OPTIMIZER SUMMARY
    # ========================================================

    optimizer_summaries.append({

        "optimizer":
            optimizer_name,

        "cv_mse_mean":
            cv_mse_mean,

        "cv_mse_std":
            cv_mse_std,

        "cv_psnr_mean":
            cv_psnr_mean,

        "cv_psnr_std":
            cv_psnr_std,

        "recommended_epochs":
            recommended_epochs,

        "test_mse":
            test_mse,

        "test_psnr":
            test_psnr
    })


    del final_model
    gc.collect()


# ============================================================
# SAVE ALL FOLD RESULTS
# ============================================================

fold_results_path = os.path.join(
    RESULTS_DIR,
    "cross_validation_fold_results.csv"
)


with open(
    fold_results_path,
    "w",
    newline=""
) as file:

    fieldnames = [
        "optimizer",
        "fold",
        "best_epoch",
        "validation_mse",
        "validation_psnr",
        "training_time_seconds"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    # CSV rows preserve each completed fold's metrics and timing; figures are supplementary evidence.
    writer.writerows(
        all_fold_results
    )


# ============================================================
# SAVE FINAL OPTIMIZER SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULTS_DIR,
    "optimizer_summary.csv"
)


with open(
    summary_path,
    "w",
    newline=""
) as file:

    fieldnames = [
        "optimizer",
        "cv_mse_mean",
        "cv_mse_std",
        "cv_psnr_mean",
        "cv_psnr_std",
        "recommended_epochs",
        "test_mse",
        "test_psnr"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    # CSV rows preserve each completed fold's metrics and timing; figures are supplementary evidence.
    writer.writerows(
        optimizer_summaries
    )


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 75)
print("FINAL IMAGE-RECOVERY RESULTS")
print("=" * 75)

print()

print(
    f"Noisy baseline MSE: "
    f"{baseline_mse:.6f}"
)

print(
    f"Noisy baseline PSNR: "
    f"{baseline_psnr:.4f} dB"
)

print()


for result in optimizer_summaries:

    print(
        f"{result['optimizer']}:"
    )

    print(
        f"  CV MSE = "
        f"{result['cv_mse_mean']:.6f} "
        f"± "
        f"{result['cv_mse_std']:.6f}"
    )

    print(
        f"  CV PSNR = "
        f"{result['cv_psnr_mean']:.4f} "
        f"± "
        f"{result['cv_psnr_std']:.4f} dB"
    )

    print(
        f"  Test MSE = "
        f"{result['test_mse']:.6f}"
    )

    print(
        f"  Test PSNR = "
        f"{result['test_psnr']:.4f} dB"
    )

    print()


print(
    "Results saved in:",
    RESULTS_DIR
)

print(
    "STAGE 3B COMPLETED SUCCESSFULLY."
)