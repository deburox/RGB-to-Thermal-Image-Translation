
import os
import csv
import tensorflow as tf

from modelss.pix2pix import (
    build_generator,
    build_discriminator,
    generator_loss,
    discriminator_loss
)

from training.data_loader import load_datasets

from config import (
    MODEL_DIR,
    EPOCHS_PIX2PIX,
    LEARNING_RATE
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

train_dataset, test_dataset = load_datasets()

print("\nDatasets loaded successfully!")


# ==========================================
# 2. BUILD GENERATOR AND DISCRIMINATOR
# ==========================================

generator = build_generator()

discriminator = build_discriminator()

print("\nGenerator and Discriminator created!")


# ==========================================
# 3. DEFINE OPTIMIZERS
# ==========================================

generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    beta_1=0.5
)

discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    beta_1=0.5
)


# ==========================================
# 4. TRAINING STEP
# ==========================================

@tf.function
def train_step(input_image, target_image):

    with tf.GradientTape() as gen_tape, \
         tf.GradientTape() as disc_tape:

        # Generate thermal image
        gen_output = generator(
            input_image,
            training=True
        )

        # Discriminator predictions
        disc_real_output = discriminator(
            [input_image, target_image],
            training=True
        )

        disc_generated_output = discriminator(
            [input_image, gen_output],
            training=True
        )

        # Calculate losses
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output,
            gen_output,
            target_image
        )

        disc_loss = discriminator_loss(
            disc_real_output,
            disc_generated_output
        )

    # Calculate gradients
    generator_gradients = gen_tape.gradient(
        gen_total_loss,
        generator.trainable_variables
    )

    discriminator_gradients = disc_tape.gradient(
        disc_loss,
        discriminator.trainable_variables
    )

    # Update generator
    generator_optimizer.apply_gradients(
        zip(
            generator_gradients,
            generator.trainable_variables
        )
    )

    # Update discriminator
    discriminator_optimizer.apply_gradients(
        zip(
            discriminator_gradients,
            discriminator.trainable_variables
        )
    )

    return (
        gen_total_loss,
        gen_gan_loss,
        gen_l1_loss,
        disc_loss
    )


# ==========================================
# 5. CREATE MODEL DIRECTORY
# ==========================================

os.makedirs(MODEL_DIR, exist_ok=True)

log_path = os.path.join(
    MODEL_DIR,
    "pix2pix_training_log.csv"
)

best_generator_path = os.path.join(
    MODEL_DIR,
    "pix2pix_generator_best.keras"
)

best_discriminator_path = os.path.join(
    MODEL_DIR,
    "pix2pix_discriminator_best.keras"
)


# ==========================================
# 6. TRAINING LOOP
# ==========================================

best_gen_loss = float("inf")

with open(log_path, "w", newline="") as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "Epoch",
        "Generator Loss",
        "GAN Loss",
        "L1 Loss",
        "Discriminator Loss"
    ])

    for epoch in range(EPOCHS_PIX2PIX):

        gen_loss_avg = tf.keras.metrics.Mean()
        gan_loss_avg = tf.keras.metrics.Mean()
        l1_loss_avg = tf.keras.metrics.Mean()
        disc_loss_avg = tf.keras.metrics.Mean()

        print(f"\nEpoch {epoch + 1}/{EPOCHS_PIX2PIX}")

        total_batches = 0

        for input_image, target_image in train_dataset:

            gen_total, gan, l1, disc = train_step(
                input_image,
                target_image
            )

            gen_loss_avg.update_state(gen_total)
            gan_loss_avg.update_state(gan)
            l1_loss_avg.update_state(l1)
            disc_loss_avg.update_state(disc)

            total_batches += 1

            if total_batches % 50 == 0:
                print(
                    f"Batch {total_batches}: "
                    f"Generator Loss = {gen_loss_avg.result():.4f}, "
                    f"Discriminator Loss = {disc_loss_avg.result():.4f}"
                )

        # Average epoch losses
        avg_gen_loss = float(gen_loss_avg.result().numpy())
        avg_gan_loss = float(gan_loss_avg.result().numpy())
        avg_l1_loss = float(l1_loss_avg.result().numpy())
        avg_disc_loss = float(disc_loss_avg.result().numpy())

        print("\nEpoch Results:")
        print(f"Generator Loss: {avg_gen_loss:.4f}")
        print(f"GAN Loss: {avg_gan_loss:.4f}")
        print(f"L1 Loss: {avg_l1_loss:.4f}")
        print(f"Discriminator Loss: {avg_disc_loss:.4f}")

        # Record losses
        writer.writerow([
            epoch + 1,
            avg_gen_loss,
            avg_gan_loss,
            avg_l1_loss,
            avg_disc_loss
        ])

        csvfile.flush()

        # Save best models based on generator loss
        if avg_gen_loss < best_gen_loss:

            best_gen_loss = avg_gen_loss

            generator.save(best_generator_path)

            discriminator.save(best_discriminator_path)

            print("\nBest Pix2Pix models saved!")


# ==========================================
# 7. SAVE FINAL MODELS
# ==========================================

generator.save(
    os.path.join(
        MODEL_DIR,
        "pix2pix_generator.keras"
    )
)

discriminator.save(
    os.path.join(
        MODEL_DIR,
        "pix2pix_discriminator.keras"
    )
)

print("\nPix2Pix training completed!")

print("Generator and Discriminator saved successfully.")
print("Training history saved successfully.")
