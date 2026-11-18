"""Historical lower-learning-rate Adam diagnostic for phantom recovery.

This standalone check uses Adam at 0.0001, MSE, ten epochs and contiguous
340/85 sample IDs. Those IDs are not the saved randomized CV fold. Printed
prediction statistics are diagnostic evidence, not a final optimizer-loss
comparison. Training is unguarded and must not be triggered by importing.
"""

import numpy as np
import tensorflow as tf

from skimage.data import shepp_logan_phantom
from skimage.transform import resize

from stage2_unet import build_unet


SEED = 42
IMAGE_SIZE = 512
TRAIN_SAMPLES = 340
VAL_SAMPLES = 85
EPOCHS = 10
BATCH_SIZE = 1

MEAN = 0.0
SIGMA = 0.1


tf.keras.utils.set_random_seed(SEED)


phantom = shepp_logan_phantom()

phantom = resize(
    phantom,
    (IMAGE_SIZE, IMAGE_SIZE),
    anti_aliasing=True
).astype(np.float32)

phantom = phantom[..., np.newaxis]

clean_image = tf.constant(
    phantom,
    dtype=tf.float32
)


def create_noisy_pair(index):

    """Map a sample ID to its repeatable noisy input and shared clean target.
    
    The stateless seed [42, sample_id] fixes that ID's Gaussian realization.
    Noise has mean zero and standard deviation 0.1 before clipping; clipping
    changes the effective error distribution near intensity limits.
    """
    index = tf.cast(index, tf.int32)

    seed = tf.stack([
        tf.constant(SEED, dtype=tf.int32),
        index
    ])

    noise = tf.random.stateless_normal(
        tf.shape(clean_image),
        seed=seed,
        mean=MEAN,
        stddev=SIGMA,
        dtype=tf.float32
    )

    noisy = tf.clip_by_value(
        clean_image + noise,
        0.0,
        1.0
    )

    return noisy, clean_image


def make_dataset(indices, training=False):

    """Create a memory-controlled pipeline of paired images from integer IDs.
    
    Only training traversal is shuffled. Pair generation is deterministic per
    ID, batches contain one sample, and prefetching is limited to one batch.
    Membership is supplied by the caller rather than selected from metrics.
    """
    ds = tf.data.Dataset.from_tensor_slices(indices)

    if training:
        ds = ds.shuffle(
            len(indices),
            seed=SEED
        )

    ds = ds.map(
        create_noisy_pair,
        num_parallel_calls=1
    )

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(1)

    return ds


# Contiguous diagnostic IDs are independent of saved CV fold membership; no split is changed here.
train_indices = np.arange(
    0,
    TRAIN_SAMPLES,
    dtype=np.int32
)

val_indices = np.arange(
    TRAIN_SAMPLES,
    TRAIN_SAMPLES + VAL_SAMPLES,
    dtype=np.int32
)


train_ds = make_dataset(
    train_indices,
    training=True
)

val_ds = make_dataset(
    val_indices,
    training=False
)


def psnr_metric(y_true, y_pred):

    """Average per-image PSNR using a maximum normalized intensity of one.
    
    PSNR is measured in decibels. Averaging image PSNR is generally different
    from computing PSNR from the dataset's average MSE.
    """
    return tf.reduce_mean(
        tf.image.psnr(
            y_true,
            y_pred,
            max_val=1.0
        )
    )


model = build_unet(
    input_shape=(512, 512, 1)
)


# IMPORTANT:
# lower learning rate than previous Adam run
# Original lower-rate diagnostic (0.0001); preserve it rather than replace it with a screened rate.
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0001
)


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


# Top-level diagnostic training is retained as source only and is not invoked for submission preparation.
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1
)


results = model.evaluate(
    val_ds,
    verbose=0,
    return_dict=True
)


print()
print("ADAM DIAGNOSTIC RESULTS")
print("-----------------------")

print(
    "Validation MSE:",
    results["mse"]
)

print(
    "Validation PSNR:",
    results["psnr_metric"]
)


# Inspect one validation prediction's min/max/mean to diagnose output saturation.
for noisy_batch, clean_batch in val_ds.take(1):

    prediction = model.predict(
        noisy_batch,
        verbose=0
    )


prediction = prediction[0]

print()
print("Prediction statistics:")
print("Minimum:", prediction.min())
print("Maximum:", prediction.max())
print("Mean:", prediction.mean())

print()
print("Clean image statistics:")
print("Minimum:", phantom.min())
print("Maximum:", phantom.max())
print("Mean:", phantom.mean())