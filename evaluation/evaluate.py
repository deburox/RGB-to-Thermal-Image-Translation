
import os
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from training.data_loader import load_datasets

from config import MODEL_DIR, RESULTS_DIR


# ==========================================
# 1. LOAD TEST DATASET
# ==========================================

print("Loading test dataset...")

train_dataset, test_dataset = load_datasets()

print("Test dataset loaded successfully!")


# ==========================================
# 2. LOAD TRAINED MODELS
# ==========================================

print("\nLoading trained models...")

unet = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "unet_best.keras")
)

pix2pix = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "pix2pix_generator_best.keras")
)

print("Both models loaded successfully!")


# ==========================================
# 3. CREATE RESULTS DIRECTORIES
# ==========================================

os.makedirs(RESULTS_DIR, exist_ok=True)

comparison_dir = os.path.join(
    RESULTS_DIR,
    "visual_comparisons"
)

os.makedirs(comparison_dir, exist_ok=True)


# ==========================================
# 4. METRIC CALCULATION FUNCTION
# ==========================================

def calculate_metrics(actual, predicted):

    # Convert images from [-1, 1] to [0, 1]
    actual = (actual + 1.0) / 2.0
    predicted = (predicted + 1.0) / 2.0

    # Keep pixel values within valid range
    actual = np.clip(actual, 0, 1)
    predicted = np.clip(predicted, 0, 1)

    # Mean Absolute Error
    mae = np.mean(np.abs(actual - predicted))

    # Root Mean Squared Error
    rmse = np.sqrt(
        np.mean((actual - predicted) ** 2)
    )

    # Peak Signal-to-Noise Ratio
    psnr = peak_signal_noise_ratio(
        actual,
        predicted,
        data_range=1.0
    )

    # Structural Similarity Index
    ssim = structural_similarity(
        actual,
        predicted,
        data_range=1.0
    )

    return mae, rmse, psnr, ssim


# ==========================================
# 5. EVALUATE BOTH MODELS
# ==========================================

unet_metrics = []
pix2pix_metrics = []

sample_count = 0

print("\nStarting model evaluation...")


for rgb_batch, thermal_batch in test_dataset:

    # Generate U-Net predictions
    unet_predictions = unet(
        rgb_batch,
        training=False
    ).numpy()

    # Generate Pix2Pix predictions
    pix2pix_predictions = pix2pix(
        rgb_batch,
        training=False
    ).numpy()

    rgb_images = rgb_batch.numpy()
    actual_images = thermal_batch.numpy()

    # Evaluate each image in the batch
    for i in range(len(rgb_images)):

        actual = actual_images[i, :, :, 0]

        unet_pred = unet_predictions[i, :, :, 0]

        pix2pix_pred = pix2pix_predictions[i, :, :, 0]

        # Calculate U-Net metrics
        unet_result = calculate_metrics(
            actual,
            unet_pred
        )

        unet_metrics.append(unet_result)

        # Calculate Pix2Pix metrics
        pix2pix_result = calculate_metrics(
            actual,
            pix2pix_pred
        )

        pix2pix_metrics.append(pix2pix_result)

        # ==================================
        # SAVE FIRST 5 VISUAL COMPARISONS
        # ==================================

        if sample_count < 5:

            rgb_display = np.clip(
                (rgb_images[i] + 1.0) / 2.0,
                0,
                1
            )

            actual_display = np.clip(
                (actual + 1.0) / 2.0,
                0,
                1
            )

            unet_display = np.clip(
                (unet_pred + 1.0) / 2.0,
                0,
                1
            )

            pix2pix_display = np.clip(
                (pix2pix_pred + 1.0) / 2.0,
                0,
                1
            )

            fig, axes = plt.subplots(
                1,
                4,
                figsize=(16, 5)
            )

            axes[0].imshow(rgb_display)
            axes[0].set_title("RGB Input")

            axes[1].imshow(
                actual_display,
                cmap="gray",
                vmin=0,
                vmax=1
            )
            axes[1].set_title("Actual Thermal")

            axes[2].imshow(
                unet_display,
                cmap="gray",
                vmin=0,
                vmax=1
            )
            axes[2].set_title("U-Net Prediction")

            axes[3].imshow(
                pix2pix_display,
                cmap="gray",
                vmin=0,
                vmax=1
            )
            axes[3].set_title("Pix2Pix Prediction")

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    comparison_dir,
                    f"comparison_{sample_count + 1}.png"
                ),
                dpi=150,
                bbox_inches="tight"
            )

            plt.close(fig)

        sample_count += 1

    print(
        f"Evaluated {sample_count} test images..."
    )


# ==========================================
# 6. CALCULATE AVERAGE METRICS
# ==========================================

unet_average = np.mean(
    np.array(unet_metrics),
    axis=0
)

pix2pix_average = np.mean(
    np.array(pix2pix_metrics),
    axis=0
)


# ==========================================
# 7. DISPLAY PERFORMANCE RESULTS
# ==========================================

metric_names = [
    "MAE",
    "RMSE",
    "PSNR",
    "SSIM"
]

print("\n===================================")
print("       MODEL PERFORMANCE")
print("===================================")

print(f"Total test images: {sample_count}")

print("\nU-Net Results:")

for name, value in zip(
    metric_names,
    unet_average
):
    print(f"{name}: {value:.4f}")


print("\nPix2Pix Results:")

for name, value in zip(
    metric_names,
    pix2pix_average
):
    print(f"{name}: {value:.4f}")


# ==========================================
# 8. SAVE METRICS TO CSV
# ==========================================

csv_path = os.path.join(
    RESULTS_DIR,
    "model_comparison.csv"
)

with open(csv_path, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Model",
        "MAE",
        "RMSE",
        "PSNR",
        "SSIM"
    ])

    writer.writerow([
        "U-Net",
        *unet_average
    ])

    writer.writerow([
        "Pix2Pix",
        *pix2pix_average
    ])


# ==========================================
# 9. FINAL MESSAGE
# ==========================================

print("\n===================================")
print("Evaluation completed successfully!")
print("===================================")

print("\nMetrics saved to:")
print(csv_path)

print("\nVisual comparisons saved to:")
print(comparison_dir)

print("\nTotal images evaluated:", sample_count)
