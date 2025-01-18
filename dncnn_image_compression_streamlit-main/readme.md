# DnCNN for Image Denoising

This project involves using **DnCNN**, a deep learning-based denoising network, to improve the quality of compressed and noisy images. The primary focus is on denoising images from the **DIV2K dataset** and testing its performance on unknown noise datasets, including **DIV2k**.

---

## Team Members
Deep Das - 2201AI55
Mridul Kumar - 2201AI52
Shubhanshi - 2201AI53

## Project Overview

### What is DnCNN?
DnCNN, introduced in 2017, is a convolutional neural network designed to remove noise from images. It is widely used for denoising Gaussian and Poisson noise and can be applied to various noise types, such as:
- Noisy transmission channels
- Low-quality older cameras
- Lossy compression techniques (e.g., JPEG compression)

### Dataset
- **DIV2K Dataset**: Originally created for image denoising challenges, this dataset contains both high-quality and low-quality images.
- **Unknown Noise Dataset (DIV2)**: Used to evaluate the generalization ability of the model for real-world unknown noise scenarios.

---

## Model Architecture

The model is a modified **17-layer DnCNN architecture**:
1. **Core Features**:
   - Batch Normalization and Residual Learning.
   - ReLU activation function in hidden layers.
   - Takes the **Y channel of YCbCr images** as input.
   - Input patches of size **50x50 pixels**.
2. **Loss Function**: Mean Squared Error (MSE).
3. **Optimizer**: Adam Optimizer.
4. **Regularization**: 
   - **L2 regularization** to combat overfitting.
5. **Data Augmentation**:
   - Flipping and rotation at 8 different random angles for better noise generalization.
6. **Reduced Layers & Batch Size**:
   - To handle large datasets and limited GPU resources, the original DnCNN was reduced in complexity:
     - Batch size: **32**

---

## Implementation Details

1. **Preprocessing**:
   - Extract the Y channel from YCbCr images.
   - Create **50x50 image patches** for training.
   - Apply data augmentation (flipping & rotation).

2. **Training**:
   - Used **training and validation datasets** from the DIV2K dataset.
   - Incorporated L2 regularization to reduce overfitting in the deep network.
   - Batch size and layers were optimized to fit the GPU's capabilities.

3. **Performance Metrics**:
   - **Peak Signal-to-Noise Ratio (PSNR)**: Improved from **29.3 dB** to **30 dB**.
   - **Structural Similarity Index Measure (SSIM)**: Improved from **0.72** to **0.75**.

---

## Results

Our model effectively denoises compressed and noisy images, achieving noticeable improvements in both PSNR and SSIM. These metrics validate the effectiveness of the DnCNN model for denoising compressed images and its ability to generalize to unknown noise datasets.

---

## How to Run the Project

### Prerequisites
- Python 3.8 or higher
- TensorFlow/Keras
- GPU with sufficient memory (optional but recommended for faster training)

### Steps to Run
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
