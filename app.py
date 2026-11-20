
import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from io import BytesIO

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from config import MODEL_DIR, THERMAL_TEST_DIR


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="RGB to Thermal Translator",
    page_icon="🌡️",
    layout="wide"
)

st.title("RGB-to-Thermal Image Translation")

st.write(
    "Generate thermal infrared representations from "
    "visible RGB images using Deep Learning."
)

st.divider()


# ==========================================
# 2. LOAD TRAINED MODELS
# ==========================================

@st.cache_resource
def load_models():

    unet_path = os.path.join(
        MODEL_DIR,
        "unet_best.keras"
    )

    pix2pix_path = os.path.join(
        MODEL_DIR,
        "pix2pix_generator_best.keras"
    )

    if not os.path.exists(unet_path):
        raise FileNotFoundError(
            f"U-Net model not found: {unet_path}"
        )

    if not os.path.exists(pix2pix_path):
        raise FileNotFoundError(
            f"Pix2Pix model not found: {pix2pix_path}"
        )

    unet = tf.keras.models.load_model(
        unet_path,
        compile=False
    )

    pix2pix = tf.keras.models.load_model(
        pix2pix_path,
        compile=False
    )

    return unet, pix2pix


# ==========================================
# 3. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(uploaded_file):

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image = np.array(image)

    image = cv2.resize(
        image,
        (320, 256)
    )

    image = image.astype(np.float32)

    # Normalize to [-1, 1]
    image = image / 127.5 - 1.0

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ==========================================
# 4. PROCESS MODEL OUTPUT
# ==========================================

def prepare_output(prediction):

    # Convert from [-1, 1] to [0, 1]
    output = (prediction + 1.0) / 2.0

    output = np.clip(
        output,
        0,
        1
    )

    return output.squeeze().astype(np.float32)


# ==========================================
# 5. FIND ACTUAL THERMAL IMAGE
# ==========================================

def get_actual_thermal(filename):

    thermal_path = os.path.join(
        THERMAL_TEST_DIR,
        filename
    )

    if not os.path.exists(thermal_path):
        return None

    thermal_image = cv2.imread(
        thermal_path,
        cv2.IMREAD_GRAYSCALE
    )

    if thermal_image is None:
        return None

    thermal_image = cv2.resize(
        thermal_image,
        (320, 256)
    )

    thermal_image = thermal_image.astype(
        np.float32
    ) / 255.0

    return thermal_image


# ==========================================
# 6. CALCULATE EVALUATION METRICS
# ==========================================

def calculate_metrics(actual, predicted):

    mae = np.mean(
        np.abs(actual - predicted)
    )

    rmse = np.sqrt(
        np.mean((actual - predicted) ** 2)
    )

    psnr = peak_signal_noise_ratio(
        actual,
        predicted,
        data_range=1.0
    )

    ssim = structural_similarity(
        actual,
        predicted,
        data_range=1.0
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "PSNR": float(psnr),
        "SSIM": float(ssim)
    }


# ==========================================
# 7. CONVERT IMAGE TO DOWNLOADABLE PNG
# ==========================================

def image_to_bytes(image):

    image_uint8 = (
        np.clip(image, 0, 1) * 255
    ).astype(np.uint8)

    pil_image = Image.fromarray(
        image_uint8
    )

    buffer = BytesIO()

    pil_image.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


# ==========================================
# 8. CREATE COMPARISON IMAGE
# ==========================================

def create_comparison_image(
    actual,
    unet_output,
    pix2pix_output
):

    images = [
        actual,
        unet_output,
        pix2pix_output
    ]

    titles = [
        "Actual Thermal",
        "U-Net Prediction",
        "Pix2Pix Prediction"
    ]

    width = 320
    height = 256
    title_height = 40

    canvas = Image.new(
        "RGB",
        (width * 3, height + title_height),
        "white"
    )

    for index, (image, title) in enumerate(
        zip(images, titles)
    ):

        image_uint8 = (
            np.clip(image, 0, 1) * 255
        ).astype(np.uint8)

        pil_image = Image.fromarray(
            image_uint8
        ).convert("RGB")

        x_position = index * width

        canvas.paste(
            pil_image,
            (x_position, title_height)
        )

        # Add title
        from PIL import ImageDraw

        draw = ImageDraw.Draw(canvas)

        draw.text(
            (x_position + 10, 12),
            title,
            fill="black"
        )

    buffer = BytesIO()

    canvas.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


# ==========================================
# 9. IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload an RGB image from the LLVIP test dataset",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# 10. DISPLAY ORIGINAL IMAGE
# ==========================================

if uploaded_file is not None:

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Original RGB Image")

    st.image(
        original_image,
        caption="Uploaded RGB Image",
        width=450
    )

    st.divider()


    # ======================================
    # 11. GENERATE PREDICTIONS
    # ======================================

    if st.button(
        "Generate Thermal Images",
        type="primary"
    ):

        try:

            with st.spinner(
                "Generating thermal predictions..."
            ):

                # Load models
                unet, pix2pix = load_models()

                # Preprocess RGB image
                input_image = preprocess_image(
                    uploaded_file
                )

                # U-Net prediction
                unet_prediction = unet.predict(
                    input_image,
                    verbose=0
                )

                # Pix2Pix prediction
                pix2pix_prediction = pix2pix.predict(
                    input_image,
                    verbose=0
                )

                # Prepare predictions
                unet_output = prepare_output(
                    unet_prediction
                )

                pix2pix_output = prepare_output(
                    pix2pix_prediction
                )

                # Find ground truth
                actual_thermal = get_actual_thermal(
                    uploaded_file.name
                )


            # ==================================
            # 12. DISPLAY IMAGE COMPARISON
            # ==================================

            st.divider()

            st.subheader(
                "Thermal Image Comparison"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.markdown("### Actual Thermal")

                if actual_thermal is not None:

                    st.image(
                        actual_thermal,
                        caption="Ground Truth",
                        clamp=True,
                        width="stretch"
                    )

                else:

                    st.warning(
                        "Matching thermal image not found."
                    )

            with col2:

                st.markdown("### U-Net")

                st.image(
                    unet_output,
                    caption="U-Net Prediction",
                    clamp=True,
                    width="stretch"
                )

            with col3:

                st.markdown("### Pix2Pix")

                st.image(
                    pix2pix_output,
                    caption="Pix2Pix Prediction",
                    clamp=True,
                    width="stretch"
                )


            # ==================================
            # 13. DOWNLOAD GENERATED IMAGES
            # ==================================

            st.divider()

            st.subheader(
                "Download Generated Images"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.download_button(
                    label="Download U-Net Prediction",
                    data=image_to_bytes(unet_output),
                    file_name="unet_prediction.png",
                    mime="image/png",
                    width="stretch"
                )

            with col2:

                st.download_button(
                    label="Download Pix2Pix Prediction",
                    data=image_to_bytes(pix2pix_output),
                    file_name="pix2pix_prediction.png",
                    mime="image/png",
                    width="stretch"
                )


            # ==================================
            # 14. METRIC CALCULATION
            # ==================================

            if actual_thermal is not None:

                st.divider()

                st.subheader(
                    "Evaluation Metrics"
                )

                unet_metrics = calculate_metrics(
                    actual_thermal,
                    unet_output
                )

                pix2pix_metrics = calculate_metrics(
                    actual_thermal,
                    pix2pix_output
                )

                metric_table = {
                    "Metric": [
                        "MAE",
                        "RMSE",
                        "PSNR (dB)",
                        "SSIM"
                    ],

                    "U-Net": [
                        unet_metrics["MAE"],
                        unet_metrics["RMSE"],
                        unet_metrics["PSNR"],
                        unet_metrics["SSIM"]
                    ],

                    "Pix2Pix": [
                        pix2pix_metrics["MAE"],
                        pix2pix_metrics["RMSE"],
                        pix2pix_metrics["PSNR"],
                        pix2pix_metrics["SSIM"]
                    ]
                }

                st.dataframe(
                    metric_table,
                    hide_index=True,
                    width="stretch"
                )

                st.caption(
                    "MAE and RMSE: lower is better. "
                    "PSNR and SSIM: higher is better."
                )

                # Download comparison
                comparison_bytes = create_comparison_image(
                    actual_thermal,
                    unet_output,
                    pix2pix_output
                )

                st.download_button(
                    label="Download Complete Comparison",
                    data=comparison_bytes,
                    file_name="thermal_comparison.png",
                    mime="image/png",
                    width="stretch"
                )

            else:

                st.info(
                    "Evaluation metrics require the matching "
                    "actual thermal image from the LLVIP test set."
                )


            st.success(
                "Thermal image generation completed!"
            )


        except Exception as e:

            st.error(
                f"An error occurred: {e}"
            )


# ==========================================
# 15. PROJECT INFORMATION
# ==========================================

st.divider()

st.subheader("About the Project")

st.write(
    """
    This project performs RGB-to-thermal image translation
    using paired visible and infrared images from the
    LLVIP dataset.

    Two deep learning algorithms are implemented:

    1. U-Net: Encoder-decoder convolutional neural network.
    2. Pix2Pix: Conditional Generative Adversarial Network.

    The models generate estimated grayscale infrared
    representations, not calibrated temperature readings.
    """
)
