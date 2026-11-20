import sys
import os
import cv2

from preprocessing.image_processing import demonstrate_processing
from config import RGB_TRAIN_DIR

# Get one RGB image
image_files = sorted(os.listdir(RGB_TRAIN_DIR))

image_path = os.path.join(RGB_TRAIN_DIR, image_files[0])

# Read image using OpenCV
image = cv2.imread(image_path)

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply preprocessing techniques
demonstrate_processing(image)