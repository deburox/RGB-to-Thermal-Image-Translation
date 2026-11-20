import os
from config import (
    RGB_TRAIN_DIR,
    THERMAL_TRAIN_DIR,
    RGB_TEST_DIR,
    THERMAL_TEST_DIR
)

folders = {
    "RGB Training": RGB_TRAIN_DIR,
    "Thermal Training": THERMAL_TRAIN_DIR,
    "RGB Testing": RGB_TEST_DIR,
    "Thermal Testing": THERMAL_TEST_DIR
}

for name, path in folders.items():
    if os.path.exists(path):
        images = [
            f for f in os.listdir(path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        print(name, ":", len(images))
    else:
        print(name, ": Folder not found")