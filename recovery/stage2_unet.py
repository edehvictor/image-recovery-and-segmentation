"""Compact U-Net architecture for single-channel image recovery.

The three-level encoder, bottleneck and skip-connected decoder map a noisy
512 x 512 x 1 phantom to a bounded recovered image of the same shape. This
module defines the shared architecture; it does not establish that a medical
segmentation experiment has been trained or evaluated.
"""

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    Concatenate
)


def build_unet(input_shape=(512, 512, 1)):

    """Build the unchanged encoder--decoder with spatially aligned skip connections.
    
    Each block contains two same-padded 3 x 3 ReLU convolutions. Pooling reduces
    resolution while 8, 16, 32 and 64 filters increase representation capacity.
    Transposed convolutions restore resolution; concatenation carries encoder
    features into the decoder. A 1 x 1 sigmoid convolution returns one channel
    in [0, 1]. The default input and output are 512 x 512 x 1.
    """
    inputs = Input(shape=input_shape)

    # Encoder level 1
    c1 = Conv2D(8, (3, 3), activation="relu", padding="same")(inputs)
    c1 = Conv2D(8, (3, 3), activation="relu", padding="same")(c1)
    # Downsample spatial resolution by two while preserving learned channel features.
    p1 = MaxPooling2D((2, 2))(c1)

    # Encoder level 2
    c2 = Conv2D(16, (3, 3), activation="relu", padding="same")(p1)
    c2 = Conv2D(16, (3, 3), activation="relu", padding="same")(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Encoder level 3
    c3 = Conv2D(32, (3, 3), activation="relu", padding="same")(p2)
    c3 = Conv2D(32, (3, 3), activation="relu", padding="same")(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    # The 64-channel bottleneck represents the coarsest spatial scale (64 x 64 for the default input).
    b = Conv2D(64, (3, 3), activation="relu", padding="same")(p3)
    # The 64-channel bottleneck represents the coarsest spatial scale (64 x 64 for the default input).
    b = Conv2D(64, (3, 3), activation="relu", padding="same")(b)

    # Decoder level 3
    u3 = Conv2DTranspose(
        32,
        (2, 2),
        strides=(2, 2),
        padding="same"
    )(b)

    # Skip connection: join upsampled decoder features with the matching encoder resolution.
    u3 = Concatenate()([u3, c3])

    c4 = Conv2D(32, (3, 3), activation="relu", padding="same")(u3)
    c4 = Conv2D(32, (3, 3), activation="relu", padding="same")(c4)

    # Decoder level 2
    u2 = Conv2DTranspose(
        16,
        (2, 2),
        strides=(2, 2),
        padding="same"
    )(c4)

    # Recover intermediate spatial detail using the level-2 encoder features.
    u2 = Concatenate()([u2, c2])

    c5 = Conv2D(16, (3, 3), activation="relu", padding="same")(u2)
    c5 = Conv2D(16, (3, 3), activation="relu", padding="same")(c5)

    # Decoder level 1
    u1 = Conv2DTranspose(
        8,
        (2, 2),
        strides=(2, 2),
        padding="same"
    )(c5)

    # Restore full-resolution detail using the level-1 encoder features.
    u1 = Concatenate()([u1, c1])

    c6 = Conv2D(8, (3, 3), activation="relu", padding="same")(u1)
    c6 = Conv2D(8, (3, 3), activation="relu", padding="same")(c6)

    # Output
    # A single sigmoid channel predicts a bounded recovered intensity at every pixel.
    outputs = Conv2D(
        1,
        (1, 1),
        activation="sigmoid"
    )(c6)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Simplified_UNet"
    )

    return model

# The architecture is safe to import; its summary demonstration runs only as a script.
if __name__ == "__main__":
    model = build_unet()
    model.summary()