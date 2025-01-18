import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_absolute_error
from PIL import Image
import pandas as pd

# Set Streamlit page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="DnCNN Image Denoising Web App",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Load the trained DnCNN model
@st.cache_resource
def load_dncnn_model():
    return load_model('dncnn_model_luminance_final.keras')

model = load_dncnn_model()

# Title and description
st.title('✨ DnCNN Image Denoising Web App')
st.markdown("""
This web application uses a **Denoising Convolutional Neural Network (DnCNN)** to remove noise from images. 
Upload a noisy image to view the denoised result along with optional evaluation metrics.
""")

# Sidebar for uploading images
st.sidebar.header("📂 Upload Images")

# Upload the noisy image
noisy_file = st.sidebar.file_uploader("Upload a Noisy Image", type=["png", "jpg", "jpeg"])
# Optionally upload the original clean image
original_file = st.sidebar.file_uploader("Upload the Clean (Ground Truth) Image (Optional)", type=["png", "jpg", "jpeg"])

# Main content: Process the uploaded images
if noisy_file is not None:
    # Read the noisy image
    noisy_img = Image.open(noisy_file).convert('RGB')
    noisy_img_np = np.array(noisy_img)

    # Convert to YCbCr and extract Y channel
    noisy_img_ycbcr = cv2.cvtColor(noisy_img_np, cv2.COLOR_RGB2YCrCb)
    noisy_img_y = noisy_img_ycbcr[:, :, 0].astype('float32') / 255.0

    # Get original image dimensions
    h_original, w_original = noisy_img_y.shape

    # Pad the image to be a multiple of 50
    h_pad = (50 - h_original % 50) % 50
    w_pad = (50 - w_original % 50) % 50

    noisy_img_y_padded = np.pad(noisy_img_y, ((0, h_pad), (0, w_pad)), 'reflect')

    h_padded, w_padded = noisy_img_y_padded.shape

    # Process the image in patches
    denoised_img_y_padded = np.zeros_like(noisy_img_y_padded)
    for y in range(0, h_padded, 50):
        for x in range(0, w_padded, 50):
            noisy_patch = noisy_img_y_padded[y:y+50, x:x+50]
            noisy_patch = np.expand_dims(noisy_patch, axis=(0, -1))  # Shape: (1, 50, 50, 1)

            # Predict the denoised patch
            denoised_patch = model.predict(noisy_patch, verbose=0)[0, :, :, 0]
            denoised_patch = np.clip(denoised_patch, 0, 1)

            denoised_img_y_padded[y:y+50, x:x+50] = denoised_patch

    # Remove the padding
    denoised_img_y = denoised_img_y_padded[:h_original, :w_original]

    # Convert back to uint8
    denoised_img_y_uint8 = (denoised_img_y * 255).astype('uint8')

    # Merge with original Cb and Cr channels
    denoised_img_ycbcr = cv2.merge((denoised_img_y_uint8,
                                     noisy_img_ycbcr[:h_original, :w_original, 1],
                                     noisy_img_ycbcr[:h_original, :w_original, 2]))

    # Convert back to RGB
    denoised_img_rgb = cv2.cvtColor(denoised_img_ycbcr, cv2.COLOR_YCrCb2RGB)

    # Display the images side by side
    st.subheader("🔍 Results")
    col1, col2 = st.columns(2)
    with col1:
        st.image(noisy_img, caption='Original Noisy Image', use_container_width=True)
    with col2:
        st.image(denoised_img_rgb, caption='Denoised Image', use_container_width=True)

    # If original clean image is uploaded, compute evaluation metrics
    if original_file is not None:
        original_img = Image.open(original_file).convert('RGB')
        original_img_np = np.array(original_img)

        # Convert original image to Y channel
        original_img_ycbcr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2YCrCb)
        original_img_y = original_img_ycbcr[:, :, 0].astype('float32') / 255.0

        # Resize if dimensions do not match
        if original_img_y.shape != denoised_img_y.shape:
            original_img_y = cv2.resize(original_img_y, (denoised_img_y.shape[1], denoised_img_y.shape[0]))

        # Calculate evaluation metrics
        mse_denoised = np.mean((original_img_y - denoised_img_y) ** 2)
        psnr_denoised = 20 * np.log10(1.0 / np.sqrt(mse_denoised))
        ssim_denoised = ssim(original_img_y, denoised_img_y, data_range=1.0)
        mae_denoised = mean_absolute_error(original_img_y.flatten(), denoised_img_y.flatten())

        mse_noisy = np.mean((original_img_y - noisy_img_y) ** 2)
        psnr_noisy = 20 * np.log10(1.0 / np.sqrt(mse_noisy))
        ssim_noisy = ssim(original_img_y, noisy_img_y, data_range=1.0)
        mae_noisy = mean_absolute_error(original_img_y.flatten(), noisy_img_y.flatten())

        # Display evaluation metrics in a table
        metrics_data = {
            "Metric": ["PSNR (dB)", "SSIM", "MSE", "MAE"],
            "Noisy vs Clean": [f"{psnr_noisy:.2f}", f"{ssim_noisy:.4f}", f"{mse_noisy:.6f}", f"{mae_noisy:.6f}"],
            "Denoised vs Clean": [f"{psnr_denoised:.2f}", f"{ssim_denoised:.4f}", f"{mse_denoised:.6f}", f"{mae_denoised:.6f}"],
        }
        metrics_df = pd.DataFrame(metrics_data)

        st.subheader("📊 Evaluation Metrics")
        st.dataframe(metrics_df.style.set_properties(**{'text-align': 'center'}).set_table_styles([
            dict(selector="th", props=[("text-align", "center")])
        ]))

else:
    st.info("Please upload a noisy image to denoise.")
