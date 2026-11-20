
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR


# ==========================================
# 1. LOAD MODEL COMPARISON RESULTS
# ==========================================

csv_path = os.path.join(
    RESULTS_DIR,
    "model_comparison.csv"
)

models = []
mae = []
rmse = []
psnr = []
ssim = []

with open(csv_path, "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        models.append(row["Model"])
        mae.append(float(row["MAE"]))
        rmse.append(float(row["RMSE"]))
        psnr.append(float(row["PSNR"]))
        ssim.append(float(row["SSIM"]))


# ==========================================
# 2. CREATE GRAPH DIRECTORY
# ==========================================

graph_dir = os.path.join(
    RESULTS_DIR,
    "graphs"
)

os.makedirs(graph_dir, exist_ok=True)


# ==========================================
# 3. GRAPH PLOTTING FUNCTION
# ==========================================

def plot_comparison(
    values,
    metric_name,
    ylabel,
    filename,
    lower_is_better
):

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.bar(
        models,
        values,
        width=0.5
    )

    ax.set_title(
        f"U-Net vs Pix2Pix - {metric_name}",
        fontsize=16,
        pad=15
    )

    ax.set_xlabel(
        "Machine Learning Models",
        fontsize=12
    )

    ax.set_ylabel(
        ylabel,
        fontsize=12,
        labelpad=10
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    ax.set_axisbelow(True)

    # Add numerical values above bars
    for bar, value in zip(bars, values):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=12
        )

    # Identify better-performing model
    if lower_is_better:
        best_index = int(np.argmin(values))
    else:
        best_index = int(np.argmax(values))

    fig.text(
        0.5,
        0.02,
        f"Better performance: {models[best_index]}",
        ha="center",
        fontsize=12
    )

    # Improve layout and prevent label clipping
    fig.subplots_adjust(
        left=0.16,
        right=0.96,
        bottom=0.15,
        top=0.88
    )

    # Save graph
    plt.savefig(
        os.path.join(graph_dir, filename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close(fig)


# ==========================================
# 4. MAE COMPARISON
# ==========================================

plot_comparison(
    mae,
    "Mean Absolute Error (MAE)",
    "MAE",
    "mae_comparison.png",
    True
)


# ==========================================
# 5. RMSE COMPARISON
# ==========================================

plot_comparison(
    rmse,
    "Root Mean Squared Error (RMSE)",
    "RMSE",
    "rmse_comparison.png",
    True
)


# ==========================================
# 6. PSNR COMPARISON
# ==========================================

plot_comparison(
    psnr,
    "Peak Signal-to-Noise Ratio (PSNR)",
    "PSNR (dB)",
    "psnr_comparison.png",
    False
)


# ==========================================
# 7. SSIM COMPARISON
# ==========================================

plot_comparison(
    ssim,
    "Structural Similarity Index (SSIM)",
    "SSIM",
    "ssim_comparison.png",
    False
)


# ==========================================
# 8. COMPLETION MESSAGE
# ==========================================

print("\nAll comparison graphs generated successfully!")

print("\nGraphs saved inside:")
print(graph_dir)
