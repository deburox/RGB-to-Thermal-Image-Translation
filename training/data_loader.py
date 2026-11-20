
import os
import cv2
import numpy as np
import tensorflow as tf
import random

from config import (
    RGB_TRAIN_DIR,
    THERMAL_TRAIN_DIR,
    RGB_TEST_DIR,
    THERMAL_TEST_DIR,
    IMG_HEIGHT,
    IMG_WIDTH,
    BATCH_SIZE,
    MAX_TRAIN_IMAGES,
    MAX_TEST_IMAGES,
    RANDOM_SEED
)


# ==========================================
# LOAD AND PREPROCESS ONE IMAGE PAIR
# ==========================================

def load_image_pair(rgb_path, thermal_path):

    if isinstance(rgb_path, bytes):
        rgb_path = rgb_path.decode("utf-8")

    if isinstance(thermal_path, bytes):
        thermal_path = thermal_path.decode("utf-8")

    rgb_image = cv2.imread(rgb_path)
    thermal_image = cv2.imread(
        thermal_path,
        cv2.IMREAD_GRAYSCALE
    )

    if rgb_image is None or thermal_image is None:
        raise ValueError("Could not read image pair.")

    # Convert RGB image from BGR to RGB
    rgb_image = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_BGR2RGB
    )

    # Resize images
    rgb_image = cv2.resize(
        rgb_image,
        (IMG_WIDTH, IMG_HEIGHT)
    )

    thermal_image = cv2.resize(
        thermal_image,
        (IMG_WIDTH, IMG_HEIGHT)
    )

    # Convert to float32
    rgb_image = rgb_image.astype(np.float32)
    thermal_image = thermal_image.astype(np.float32)

    # Normalize pixel values to [-1, 1]
    rgb_image = (rgb_image / 127.5) - 1.0
    thermal_image = (thermal_image / 127.5) - 1.0

    # Add channel dimension to thermal image
    thermal_image = np.expand_dims(
        thermal_image,
        axis=-1
    )

    return rgb_image, thermal_image


# ==========================================
# GET MATCHING IMAGE PAIRS
# ==========================================

def get_image_pairs(
    rgb_dir,
    thermal_dir,
    max_images,
    seed
):

    rgb_files = sorted([
        f for f in os.listdir(rgb_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    thermal_files = sorted([
        f for f in os.listdir(thermal_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    # Check whether filenames match
    if rgb_files != thermal_files:
        raise ValueError(
            "RGB and thermal filenames do not match!"
        )

    if len(rgb_files) < max_images:
        raise ValueError(
            f"Expected {max_images} images, "
            f"but found only {len(rgb_files)}."
        )

    # Select a reproducible random subset
    rng = random.Random(seed)

    selected_files = rng.sample(
        rgb_files,
        max_images
    )

    selected_files.sort()

    rgb_paths = [
        os.path.join(rgb_dir, f)
        for f in selected_files
    ]

    thermal_paths = [
        os.path.join(thermal_dir, f)
        for f in selected_files
    ]

    return rgb_paths, thermal_paths


# ==========================================
# CREATE TENSORFLOW DATASET
# ==========================================

def create_dataset(
    rgb_dir,
    thermal_dir,
    max_images,
    seed,
    training=False
):

    rgb_paths, thermal_paths = get_image_pairs(
        rgb_dir,
        thermal_dir,
        max_images,
        seed
    )

    print(
        f"Selected {len(rgb_paths)} image pairs "
        f"from {rgb_dir}"
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (rgb_paths, thermal_paths)
    )

    # Shuffle training pairs together
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(rgb_paths),
            seed=seed,
            reshuffle_each_iteration=True
        )

    def load_pair(rgb_path, thermal_path):

        rgb, thermal = tf.numpy_function(
            load_image_pair,
            [rgb_path, thermal_path],
            [tf.float32, tf.float32]
        )

        rgb.set_shape(
            (IMG_HEIGHT, IMG_WIDTH, 3)
        )

        thermal.set_shape(
            (IMG_HEIGHT, IMG_WIDTH, 1)
        )

        return rgb, thermal

    dataset = dataset.map(
        load_pair,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ==========================================
# LOAD TRAINING AND TESTING DATASETS
# ==========================================

def load_datasets():

    train_dataset = create_dataset(
        RGB_TRAIN_DIR,
        THERMAL_TRAIN_DIR,
        MAX_TRAIN_IMAGES,
        RANDOM_SEED,
        training=True
    )

    test_dataset = create_dataset(
        RGB_TEST_DIR,
        THERMAL_TEST_DIR,
        MAX_TEST_IMAGES,
        RANDOM_SEED + 1,
        training=False
    )

    return train_dataset, test_dataset
