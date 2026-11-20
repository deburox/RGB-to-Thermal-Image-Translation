# RGB-to-Thermal Image Translation Using Conditional GAN

A deep learning project that translates visible RGB images into grayscale thermal infrared representations using paired images from the **LLVIP dataset**.

The proposed approach uses a **Pix2Pix Conditional Generative Adversarial Network (cGAN)**. A standalone U-Net model is also implemented as a baseline for performance comparison.

## Project Overview

Thermal imaging is useful in applications such as surveillance, night-time monitoring, and object detection. However, thermal cameras can be expensive.

This project explores the use of deep learning to generate estimated thermal infrared representations from visible RGB images.

The system takes an RGB image as input and generates a corresponding grayscale thermal image.

**Proposed Model:** Pix2Pix Conditional GAN  
**Baseline Model:** U-Net  
**Dataset:** LLVIP (Low-Light Visible-Infrared Paired Dataset)

## Model Architecture

### 1. Pix2Pix Conditional GAN — Proposed Model

Pix2Pix is a conditional Generative Adversarial Network that learns to translate an input image into a corresponding target image.

Its architecture consists of:

- **Generator:** U-Net-based encoder-decoder network that generates thermal images from RGB inputs.
- **Discriminator:** PatchGAN discriminator that distinguishes real thermal images from generated thermal images.
- **Adversarial Loss:** Binary cross-entropy loss.
- **Reconstruction Loss:** L1 loss, which measures the difference between generated and actual thermal images.
- **Combined Generator Loss:** Adversarial loss + 100 × L1 loss.

### 2. U-Net — Baseline Model

A standalone U-Net encoder-decoder convolutional neural network is implemented as a baseline.

It learns the mapping between visible RGB images and corresponding thermal images using reconstruction loss.

Both models are evaluated using the same test dataset.

## Dataset

This project uses the **LLVIP dataset**, which contains paired visible-light and infrared images.

Dataset directory structure:

```text
dataset/
├── visible/
│   ├── train/
│   └── test/
│
└── infrared/
    ├── train/
    └── test/
```

The current configuration uses:

| Parameter | Value |
|---|---:|
| Training image pairs | 2,000 |
| Testing image pairs | 500 |
| Image height | 256 |
| Image width | 320 |
| Batch size | 4 |
| U-Net epochs | 5 |
| Pix2Pix epochs | 5 |
| Learning rate | 0.0002 |

The dataset is not included in this repository. Download the LLVIP dataset separately and arrange the folders as shown above.

## Image Processing

The project also implements the following image processing techniques:

- **Gaussian Filtering:** Used to reduce image noise and smooth image details.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Used to improve local image contrast.

These techniques are implemented separately in the preprocessing module.

## Evaluation Metrics

The models are evaluated using four metrics:

| Metric | Purpose |
|---|---|
| MAE | Measures average absolute pixel-level error |
| RMSE | Measures the square root of average squared error |
| PSNR | Measures reconstruction quality in terms of signal-to-noise ratio |
| SSIM | Measures structural similarity between images |

## Model Comparison Results

The following results were obtained on the 500-image test subset:

| Model | MAE ↓ | RMSE ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|
| U-Net | 0.1674 | 0.2114 | 13.5357 dB | 0.3752 |
| Pix2Pix | 0.1891 | 0.2383 | 12.5309 dB | 0.2099 |

**Observation:** U-Net achieved better results than Pix2Pix across all four evaluation metrics in the current experiment. This provides a useful baseline comparison and indicates that the Pix2Pix model may require further training and tuning.

The evaluation graphs are available in:

```text
results/graphs/
```

The numerical comparison is available in:

```text
results/model_comparison.csv
```

## Streamlit Application

The project includes a Streamlit web application that allows users to:

1. Upload a visible RGB image.
2. Generate a thermal image using the U-Net model.
3. Generate a thermal image using the Pix2Pix generator.
4. Compare both generated outputs with the actual thermal image when a matching test image is available.
5. View evaluation metrics.
6. Download generated images and the comparison image.

### Run the Application

First, clone the repository:

```bash
git clone https://github.com/deburox/RGB-to-Thermal-Image-Translation.git
```

Navigate into the project directory:

```bash
cd RGB-to-Thermal-Image-Translation
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Ensure the dataset and trained model files are available in their expected directories.

Run the Streamlit application:

```bash
streamlit run app.py
```

## Project Structure

```text
RGB_THERMAL_PROJECT/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
│
├── dataset/
│
├── modelss/
│   ├── __init__.py
│   └── saved/
│
├── preprocessing/
│   └── image_processing.py
│
├── training/
│   ├── data_loader.py
│   ├── train_unet.py
│   └── train_pix2pix.py
│
├── evaluation/
│   └── evaluate.py
│
└── results/
    ├── model_comparison.csv
    └── graphs/
```

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
