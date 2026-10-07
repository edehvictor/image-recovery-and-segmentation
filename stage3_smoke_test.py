import os
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from skimage.data import shepp_logan_phantom
from skimage.transform import resize

from stage2_unet import build_unet


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

SEED = 42

IMAGE_SIZE = 512

GAUSSIAN_MEAN = 0.0
GAUSSIAN_STD = 0.1

BATCH_SIZE = 1

# This is ONLY a quick test.
# These are NOT the final experimental sample sizes.
SMOKE_TRAIN_SAMPLES = 32
SMOKE_VAL_SAMPLES = 8

EPOCHS = 1


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# RESULTS DIRECTORY
# ============================================================

os.makedirs(
    "results/stage3_smoke_test",
    exist_ok=True
)


# ============================================================
# CREATE CLEAN 512 x 512 SHEPP-LOGAN PHANTOM
# ============================================================

phantom = shepp_logan_phantom()

phantom = resize(
    phantom,
    (IMAGE_SIZE, IMAGE_SIZE),
    anti_aliasing=True
)

phantom = phantom.astype(np.float32)

# Add grayscale channel:
# (512, 512) -> (512, 512, 1)
phantom = phantom[..., np.newaxis]


print("Clean phantom shape:", phantom.shape)

print(
    "Pixel range:",
    phantom.min(),
    "to",
    phantom.max()
)


# Convert clean image to TensorFlow tensor
clean_image = tf.constant(
    phantom,
    dtype=tf.float32
)


# ============================================================
# CREATE NOISY IMAGE FROM SAMPLE INDEX
# ============================================================

def create_noisy_pair(index):

    index = tf.cast(index, tf.int32)

    # Each index gets its own deterministic random seed.
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
# CREATE DATASET
# ============================================================

def make_dataset(indices, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        indices
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

    # Keep memory usage controlled
    dataset = dataset.prefetch(1)

    return dataset


# ============================================================
# SMOKE-TEST INDICES
# ============================================================

train_indices = np.arange(
    0,
    SMOKE_TRAIN_SAMPLES,
    dtype=np.int32
)

val_indices = np.arange(
    SMOKE_TRAIN_SAMPLES,
    SMOKE_TRAIN_SAMPLES + SMOKE_VAL_SAMPLES,
    dtype=np.int32
)


train_dataset = make_dataset(
    train_indices,
    training=True
)

val_dataset = make_dataset(
    val_indices,
    training=False
)


print()
print("Smoke-test training images:",
      len(train_indices))

print("Smoke-test validation images:",
      len(val_indices))


# ============================================================
# PSNR METRIC
# ============================================================

def psnr_metric(y_true, y_pred):

    psnr_values = tf.image.psnr(
        y_true,
        y_pred,
        max_val=1.0
    )

    return tf.reduce_mean(psnr_values)


# ============================================================
# BUILD U-NET
# ============================================================

model = build_unet(
    input_shape=(512, 512, 1)
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer=optimizer,

    loss="mse",

    metrics=[
        tf.keras.metrics.MeanSquaredError(
            name="mse"
        ),

        psnr_metric
    ]
)


print()
print("Starting smoke-test training...")
print("Optimizer: Adam")
print("Loss function: MSE")
print()


# ============================================================
# TRAIN MODEL
# ============================================================

start_time = time.time()


history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    verbose=1
)


end_time = time.time()

training_time = end_time - start_time


print()
print(
    f"Training time: "
    f"{training_time:.2f} seconds"
)


# ============================================================
# VALIDATION EVALUATION
# ============================================================

validation_results = model.evaluate(
    val_dataset,
    verbose=0,
    return_dict=True
)


print()
print("VALIDATION RESULTS")

for metric_name, metric_value in validation_results.items():

    print(
        f"{metric_name}: "
        f"{metric_value:.6f}"
    )


# ============================================================
# RECOVER ONE VALIDATION IMAGE
# ============================================================

for noisy_batch, clean_batch in val_dataset.take(1):

    prediction = model.predict(
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
        prediction[0]
        .squeeze()
    )


# ============================================================
# CALCULATE SAMPLE MSE
# ============================================================

sample_mse = np.mean(

    (
        clean_example
        -
        recovered_example
    ) ** 2
)


# ============================================================
# CALCULATE SAMPLE PSNR
# ============================================================

if sample_mse > 0:

    sample_psnr = (
        10
        *
        np.log10(
            1.0 / sample_mse
        )
    )

else:

    sample_psnr = np.inf


print()

print(
    f"Sample recovered-image MSE: "
    f"{sample_mse:.6f}"
)

print(
    f"Sample recovered-image PSNR: "
    f"{sample_psnr:.4f} dB"
)


# ============================================================
# SAVE VISUAL RESULT
# ============================================================

plt.figure(
    figsize=(12, 4)
)


plt.subplot(1, 3, 1)

plt.imshow(
    clean_example,
    cmap="gray"
)

plt.title("Clean Phantom")

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    noisy_example,
    cmap="gray"
)

plt.title("Noisy Input")

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    recovered_example,
    cmap="gray"
)

plt.title("Recovered Output")

plt.axis("off")


plt.tight_layout()


plt.savefig(

    "results/stage3_smoke_test/"
    "smoke_test_recovery.png",

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# SAVE LOSS CURVE
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)


plt.xlabel("Epoch")

plt.ylabel("MSE Loss")

plt.title(
    "Smoke Test: Training and Validation Loss"
)

plt.legend()

plt.grid(True)


plt.savefig(

    "results/stage3_smoke_test/"
    "smoke_test_loss.png",

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print()
print(
    "Smoke-test figures saved successfully."
)

print(
    "STAGE 3A SMOKE TEST COMPLETED."
)