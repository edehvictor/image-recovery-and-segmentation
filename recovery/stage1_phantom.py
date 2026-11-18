"""Historical Shepp--Logan image-degradation demonstration.

The analytically defined phantom supplies a fixed, normalized grayscale target
for a controlled denoising task. Independent Gaussian perturbations produce
noisy inputs; all targets contain the same anatomy. This script materializes
500 image pairs in memory and saves two visualizations, not a persistent
dataset. Its 350/75/75 demonstration split is distinct from Stage 3's saved
425-development/75-pilot-test split. Work at module level executes if imported.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from skimage.transform import resize


# =========================================================
# STAGE ONE - PART A
# GENERATE AND DEGRADE THE SHEPP-LOGAN PHANTOM
# =========================================================

# Generate the original Shepp-Logan phantom
# Load the fixed synthetic head phantom supplied by scikit-image, not a patient image.
phantom = shepp_logan_phantom()

print("Type:", type(phantom))
print("Original shape:", phantom.shape)


# Resize the phantom to 512 x 512
# Resize with anti-aliasing and float32 conversion; no additional min--max normalization occurs.
phantom = resize(
    phantom,
    (512, 512),
    anti_aliasing=True
).astype(np.float32)

print("New shape:", phantom.shape)
print("Minimum pixel value:", phantom.min())
print("Maximum pixel value:", phantom.max())


# Gaussian noise parameters
# Gaussian parameters describe the added noise before clipping, not the clipped residual distribution.
mean = 0
sigma = 0.1


# Create a reproducible random-number generator
# A local seeded generator is reproducible; later draws depend on all earlier draws in this script.
rng = np.random.default_rng(42)


# Generate Gaussian noise
noise = rng.normal(
    loc=mean,
    scale=sigma,
    size=phantom.shape
).astype(np.float32)


# Add the noise to the clean phantom
noisy_phantom = phantom + noise


# Keep all pixel values between 0 and 1
# Bound observed intensities to [0,1]; the clean normalized phantom already lies in that range.
noisy_phantom = np.clip(
    noisy_phantom,
    0,
    1
)


# Create a folder where results will be saved
os.makedirs("results", exist_ok=True)


# Display clean and noisy phantom
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(phantom, cmap="gray")
plt.title("Clean Shepp-Logan Phantom")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(noisy_phantom, cmap="gray")
plt.title("Noisy Shepp-Logan Phantom")
plt.axis("off")

plt.tight_layout()

# Save the result
plt.savefig(
    "results/clean_and_noisy_phantom.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# STAGE ONE - PART B
# CREATE MULTIPLE NOISY SAMPLES
# =========================================================

# These 500 pairs vary only in noise realization; targets repeat the same phantom.
num_samples = 500


# Create arrays to store the images
noisy_images = np.empty(
    (num_samples, 512, 512, 1),
    dtype=np.float32
)

clean_images = np.empty(
    (num_samples, 512, 512, 1),
    dtype=np.float32
)


# Generate 500 different noisy versions of the same phantom
# Use successive generator draws; this demonstration is not Stage 3's stateless per-ID generator.
for i in range(num_samples):

    noise = rng.normal(
        loc=mean,
        scale=sigma,
        size=phantom.shape
    ).astype(np.float32)

    noisy = phantom + noise

    noisy = np.clip(
        noisy,
        0,
        1
    )

    # Store noisy image
    noisy_images[i, :, :, 0] = noisy

    # Store corresponding clean target
    clean_images[i, :, :, 0] = phantom


print("\nDataset created successfully")
print("Noisy dataset shape:", noisy_images.shape)
print("Clean dataset shape:", clean_images.shape)


# =========================================================
# SPLIT DATA INTO TRAINING, VALIDATION AND TESTING
# =========================================================

# Shuffle the positions of the 500 samples
# Demonstration-only 350/75/75 partition; it must not replace the saved revised CV membership.
indices = rng.permutation(num_samples)


# 70% training = 350 images
train_indices = indices[:350]

# 15% validation = 75 images
val_indices = indices[350:425]

# 15% testing = 75 images
test_indices = indices[425:500]


print("\nTraining samples:", len(train_indices))
print("Validation samples:", len(val_indices))
print("Testing samples:", len(test_indices))


# =========================================================
# DISPLAY ONE TRAINING EXAMPLE
# =========================================================

# Save the first selected training pair without fitting a network or saving the complete arrays.
sample_index = train_indices[0]

plt.figure(figsize=(10, 5))


# Noisy input
plt.subplot(1, 2, 1)

plt.imshow(
    noisy_images[sample_index, :, :, 0],
    cmap="gray"
)

plt.title("Training Input - Noisy Phantom")
plt.axis("off")


# Clean target
plt.subplot(1, 2, 2)

plt.imshow(
    clean_images[sample_index, :, :, 0],
    cmap="gray"
)

plt.title("Training Target - Clean Phantom")
plt.axis("off")


plt.tight_layout()


# Save the training example
plt.savefig(
    "results/stage1_training_pair.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nStage One completed successfully.")