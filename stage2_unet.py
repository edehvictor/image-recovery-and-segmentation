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

    inputs = Input(shape=input_shape)

    # Encoder level 1
    c1 = Conv2D(8, (3, 3), activation="relu", padding="same")(inputs)
    c1 = Conv2D(8, (3, 3), activation="relu", padding="same")(c1)
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
    b = Conv2D(64, (3, 3), activation="relu", padding="same")(p3)
    b = Conv2D(64, (3, 3), activation="relu", padding="same")(b)

    # Decoder level 3
    u3 = Conv2DTranspose(
        32,
        (2, 2),
        strides=(2, 2),
        padding="same"
    )(b)

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

    u1 = Concatenate()([u1, c1])

    c6 = Conv2D(8, (3, 3), activation="relu", padding="same")(u1)
    c6 = Conv2D(8, (3, 3), activation="relu", padding="same")(c6)

    # Output
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

if __name__ == "__main__":
    model = build_unet()
    model.summary()