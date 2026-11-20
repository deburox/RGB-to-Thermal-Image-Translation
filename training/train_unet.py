
import tensorflow as tf
import os

from modelss.unet import build_unet
from training.data_loader import load_datasets
from config import MODEL_DIR, EPOCHS_UNET, LEARNING_RATE


# Load datasets
train_dataset, test_dataset = load_datasets()


# Build U-Net model
unet = build_unet()


# Compile model
unet.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="mae",
    metrics=["mae"]
)


# Create model output directory
os.makedirs(MODEL_DIR, exist_ok=True)


# Save the best model during training
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "unet_best.keras"),
    monitor="loss",
    save_best_only=True,
    verbose=1
)


# Record training history
csv_logger = tf.keras.callbacks.CSVLogger(
    os.path.join(MODEL_DIR, "unet_training_log.csv")
)


# Train U-Net
history = unet.fit(
    train_dataset,
    epochs=EPOCHS_UNET,
    callbacks=[checkpoint, csv_logger]
)


# Save final model
unet.save(
    os.path.join(MODEL_DIR, "unet_model.keras")
)


print("\nU-Net training completed!")
print("Best model and final model saved successfully.")
