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