
# RGB-to-Thermal Image Translation Using Deep Learning

## 1. Project Overview

This project implements RGB-to-thermal image translation using deep learning.

It takes a visible RGB image as input and generates an estimated grayscale infrared image using two deep learning architectures: U-Net and Pix2Pix.

The project uses the LLVIP paired visible-infrared dataset.

## 2. Objectives

- Convert visible RGB images into estimated thermal infrared representations.
- Implement and compare U-Net and Pix2Pix architectures.
- Apply image-processing techniques to study image enhancement.
- Evaluate model performance using quantitative metrics.
- Develop an interactive application for real-time image translation.

## 3. Technologies Used

- Python 3.13
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-image
- Streamlit

## 4. Deep Learning Models

### U-Net
U-Net is an encoder-decoder convolutional neural network with skip connections. It learns spatial features from RGB images and generates grayscale thermal predictions.

### Pix2Pix
Pix2Pix is a conditional Generative Adversarial Network (cGAN). It consists of a generator and a PatchGAN discriminator.

The generator produces thermal images, while the discriminator distinguishes between real and generated RGB-thermal image pairs.

## 5. Image Processing Techniques

- Gaussian Filtering
- Contrast Limited Adaptive Histogram Equalization (CLAHE)

## 6. Dataset

Dataset: LLVIP (Low-Light Vision and Infrared Paired Dataset).

The dataset contains paired visible and infrared images.

Expected directory structure:

dataset/
- visible/
  - train/
  - test/
- infrared/
  - train/
  - test/

## 7. Evaluation Metrics

The models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)

Lower MAE and RMSE indicate better performance, while higher PSNR and SSIM indicate better image similarity.

## 8. Application Features

- Upload an RGB image.
- Generate thermal predictions using U-Net.
- Generate thermal predictions using Pix2Pix.
- Compare generated images with actual thermal images when paired ground truth is available.
- View evaluation metrics.
- Download generated thermal images and comparison results.

## 9. Installation and Execution

### Step 1: Clone the repository

git clone <your-repository-link>

### Step 2: Navigate to the project directory

cd RGB_THERMAL_PROJECT

### Step 3: Create a virtual environment

python3.13 -m venv .venv

### Step 4: Activate the environment

source .venv/bin/activate

### Step 5: Install dependencies

pip install -r requirements.txt

### Step 6: Run the application

streamlit run app.py

## 10. Results

The project compares U-Net and Pix2Pix using MAE, RMSE, PSNR and SSIM.

The experimental results show that U-Net performs better than Pix2Pix on the current evaluation dataset.

## 11. Limitations

- Generated images are estimated grayscale infrared representations.
- The generated images do not provide calibrated temperature measurements.
- Prediction quality depends on training data, model architecture and training duration.

## 12. Future Scope

- Improve image quality through hyperparameter tuning.
- Experiment with deeper architectures and attention mechanisms.
- Increase training dataset size.
- Explore real-time deployment on edge devices.

## 13. Author

Developed as an academic deep learning project.
