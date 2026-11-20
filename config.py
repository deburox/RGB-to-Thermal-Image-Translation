
import os

# ==========================================
# PROJECT DIRECTORY
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==========================================
# DATASET DIRECTORIES
# ==========================================

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Training directories
RGB_TRAIN_DIR = os.path.join(DATASET_DIR, "visible", "train")
THERMAL_TRAIN_DIR = os.path.join(DATASET_DIR, "infrared", "train")

# Testing directories
RGB_TEST_DIR = os.path.join(DATASET_DIR, "visible", "test")
THERMAL_TEST_DIR = os.path.join(DATASET_DIR, "infrared", "test")


# ==========================================
# DATASET SIZE
# ==========================================

MAX_TRAIN_IMAGES = 2000
MAX_TEST_IMAGES = 500


# ==========================================
# IMAGE DIMENSIONS
# ==========================================

IMG_HEIGHT = 256
IMG_WIDTH = 320


# ==========================================
# TRAINING CONFIGURATION
# ==========================================

BATCH_SIZE = 4

EPOCHS_UNET = 5
EPOCHS_PIX2PIX = 5

LEARNING_RATE = 0.0002


# ==========================================
# OUTPUT DIRECTORIES
# ==========================================

MODEL_DIR = os.path.join(BASE_DIR, "modelss", "saved")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================================
# RANDOM SEED
# ==========================================

RANDOM_SEED = 42
