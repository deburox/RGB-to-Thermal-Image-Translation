
import tensorflow as tf
from tensorflow.keras import layers, Model


# ==========================================
# 1. PIX2PIX GENERATOR
# ==========================================

def build_generator(input_shape=(256, 320, 3)):

    inputs = layers.Input(shape=input_shape)

    # Encoder
    filters = [32, 64, 128, 256, 256]

    skips = []
    x = inputs

    for f in filters:

        x = layers.Conv2D(
            f,
            4,
            strides=2,
            padding="same"
        )(x)

        x = layers.LeakyReLU(0.2)(x)

        skips.append(x)

    # Decoder
    decoder_filters = [256, 128, 64, 32]

    for f, skip in zip(
        decoder_filters,
        reversed(skips[:-1])
    ):

        x = layers.Conv2DTranspose(
            f,
            4,
            strides=2,
            padding="same"
        )(x)

        x = layers.BatchNormalization()(x)

        x = layers.ReLU()(x)

        # Skip connection
        x = layers.Concatenate()([x, skip])

    # Output layer: grayscale thermal image
    outputs = layers.Conv2DTranspose(
        1,
        4,
        strides=2,
        padding="same",
        activation="tanh"
    )(x)

    return Model(
        inputs,
        outputs,
        name="Pix2Pix_Generator"
    )


# ==========================================
# 2. PIX2PIX DISCRIMINATOR (PATCHGAN)
# ==========================================

def build_discriminator(
    input_shape=(256, 320, 3),
    target_shape=(256, 320, 1)
):

    input_image = layers.Input(shape=input_shape)

    target_image = layers.Input(shape=target_shape)

    # Combine RGB image and thermal image
    x = layers.Concatenate()(
        [input_image, target_image]
    )

    filters = [64, 128, 256, 512]

    for i, f in enumerate(filters):

        x = layers.Conv2D(
            f,
            4,
            strides=2 if i < 3 else 1,
            padding="same"
        )(x)

        if i != 0:

            x = layers.BatchNormalization()(x)

        x = layers.LeakyReLU(0.2)(x)

    # Patch-level real/fake prediction
    outputs = layers.Conv2D(
        1,
        4,
        strides=1,
        padding="same"
    )(x)

    return Model(
        [input_image, target_image],
        outputs,
        name="PatchGAN_Discriminator"
    )


# ==========================================
# 3. LOSS FUNCTIONS
# ==========================================

loss_object = tf.keras.losses.BinaryCrossentropy(
    from_logits=True
)

LAMBDA = 100


# Generator loss
def generator_loss(
    disc_generated_output,
    gen_output,
    target
):

    gan_loss = loss_object(
        tf.ones_like(disc_generated_output),
        disc_generated_output
    )

    l1_loss = tf.reduce_mean(
        tf.abs(target - gen_output)
    )

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


# Discriminator loss
def discriminator_loss(
    disc_real_output,
    disc_generated_output
):

    real_loss = loss_object(
        tf.ones_like(disc_real_output),
        disc_real_output
    )

    generated_loss = loss_object(
        tf.zeros_like(disc_generated_output),
        disc_generated_output
    )

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss
