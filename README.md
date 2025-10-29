🧠 OncoVision: Brain Tumor Detection using Deep Learning
OncoVision is an AI-powered deep learning model designed to detect and localize brain tumors from MRI scans. Built with PyTorch, it leverages a custom CNN architecture integrated with Grad-CAM visualization to highlight tumor regions and estimate tumor growth percentage.

🚀 Project Overview
Brain tumors pose serious diagnostic challenges due to subtle visual differences between healthy and affected brain tissues.
OncoVision aims to assist radiologists and researchers by providing:
➤ Accurate tumor classification (Tumor / Healthy)
➤ Visual explanation using Grad-CAM heatmaps
➤ Tumor growth estimation percentage based on image analysis
➤ This project bridges deep learning and medical imaging to support early and transparent diagnosis.

🧩 Key Features
➤ Binary Classification: Detects if a brain MRI shows a tumor or is healthy.
➤ Grad-CAM Visualization: Highlights the specific region where the tumor is detected.
➤ Tumor Growth Estimation: Calculates approximate tumor spread percentage.
➤ Custom CNN Architecture: Lightweight yet accurate convolutional neural network.
➤ Model Training Pipeline: Easily trainable and extendable for medical datasets.

🧠 Model Architecture
The model uses a Custom CNN with the following components:
➤ Convolutional + ReLU + MaxPooling layers for feature extraction
➤ Fully connected dense layers for classification
➤ Softmax layer for binary output (Tumor / Healthy)
For interpretability, Grad-CAM is applied to visualize activated regions indicating the tumor presence.
