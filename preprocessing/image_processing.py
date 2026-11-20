import cv2
import matplotlib.pyplot as plt


def apply_gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def apply_clahe(image):

    # Convert RGB to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Apply CLAHE to the L (brightness) channel
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Convert back to RGB
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def demonstrate_processing(image):

    gaussian = apply_gaussian_filter(image)
    enhanced = apply_clahe(image)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original RGB")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gaussian)
    plt.title("Gaussian Filtering")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(enhanced)
    plt.title("CLAHE Enhancement")
    plt.axis("off")

    plt.tight_layout()
    plt.show()