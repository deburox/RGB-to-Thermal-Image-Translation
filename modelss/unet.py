
from tensorflow.keras import layers, Model


def build_unet(input_shape=(256, 320, 3)):

    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D()(c3)

    # Bottleneck
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(p3)

    # Decoder
    u3 = layers.UpSampling2D()(b)
    u3 = layers.Concatenate()([u3, c3])
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(u3)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)

    u1 = layers.UpSampling2D()(c5)
    u1 = layers.Concatenate()([u1, c1])
    c6 = layers.Conv2D(32, 3, activation='relu', padding='same')(u1)

    outputs = layers.Conv2D(
        1, 1, activation='tanh'
    )(c6)

    return Model(inputs, outputs, name="U-Net")
from modelss.unet import build_unet

from modelss.pix2pix import (
    build_generator,
    build_discriminator
)

# Build U-Net
unet = build_unet()

# Build Pix2Pix models
generator = build_generator()
discriminator = build_discriminator()

print("All models built successfully!")

print("\nU-Net output shape:")
print(unet.output_shape)

print("\nGenerator output shape:")
print(generator.output_shape)

print("\nDiscriminator output shape:")
print(discriminator.output_shape)