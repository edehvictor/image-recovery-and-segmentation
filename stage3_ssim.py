import os
import numpy as np
import tensorflow as tf

from skimage.data import shepp_logan_phantom
from skimage.transform import resize

from stage2_unet import build_unet


SEED = 42
IMAGE_SIZE = 512
NUM_SAMPLES = 500
TEST_SAMPLES = 75

RESULTS_DIR = "results/stage3_cross_validation"


# ------------------------------------------------------------
# CLEAN PHANTOM
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# RECREATE EXACT SAME TEST SPLIT
# ------------------------------------------------------------

all_indices = np.arange(
    NUM_SAMPLES,
    dtype=np.int32
)

rng = np.random.default_rng(SEED)

shuffled_indices = rng.permutation(
    all_indices
)

test_indices = shuffled_indices[
    :TEST_SAMPLES
]


# ------------------------------------------------------------
# RECREATE EXACT SAME NOISY IMAGES
# ------------------------------------------------------------

def create_noisy_pair(index):

    index = tf.cast(index, tf.int32)

    sample_seed = tf.stack([
        tf.constant(SEED, dtype=tf.int32),
        index
    ])

    noise = tf.random.stateless_normal(
        shape=tf.shape(clean_image),
        seed=sample_seed,
        mean=0.0,
        stddev=0.1,
        dtype=tf.float32
    )

    noisy = clean_image + noise

    noisy = tf.clip_by_value(
        noisy,
        0.0,
        1.0
    )

    return noisy, clean_image


test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_indices)
    .map(create_noisy_pair)
    .batch(1)
)


# ------------------------------------------------------------
# BASELINE SSIM
# ------------------------------------------------------------

baseline_ssim = []


for noisy, clean in test_dataset:

    value = tf.image.ssim(
        clean,
        noisy,
        max_val=1.0
    )

    baseline_ssim.append(
        float(value.numpy()[0])
    )


print()
print(
    "Noisy baseline SSIM:",
    np.mean(baseline_ssim)
)


# ------------------------------------------------------------
# OPTIMIZER MODELS
# ------------------------------------------------------------

optimizers = [
    "sgd",
    "rmsprop",
    "adam"
]


for optimizer_name in optimizers:

    tf.keras.backend.clear_session()

    model = build_unet(
        input_shape=(512, 512, 1)
    )

    weights_path = os.path.join(
        RESULTS_DIR,
        f"{optimizer_name}_final.weights.h5"
    )

    model.load_weights(
        weights_path
    )

    ssim_values = []


    for noisy, clean in test_dataset:

        prediction = model(
            noisy,
            training=False
        )

        value = tf.image.ssim(
            clean,
            prediction,
            max_val=1.0
        )

        ssim_values.append(
            float(value.numpy()[0])
        )


    mean_ssim = np.mean(
        ssim_values
    )

    std_ssim = np.std(
        ssim_values,
        ddof=1
    )


    print()

    print(
        optimizer_name.upper(),
        "SSIM:"
    )

    print(
        f"Mean = {mean_ssim:.6f}"
    )

    print(
        f"Standard deviation = "
        f"{std_ssim:.6f}"
    )


print()
print("SSIM CALCULATION COMPLETED.")