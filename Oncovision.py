# nw.py — OncoVision (upload + file uploader inside same box)

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image, ImageStat

# ----------------------------
# CNN Model with Grad-CAM
# ----------------------------
class CNN_CAM(nn.Module):
    def __init__(self):
        super(CNN_CAM, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        self.features = x
        self.features.retain_grad()
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sig(self.fc3(x))

# ----------------------------
# Grad-CAM
# ----------------------------
def generate_cam(model, img_tensor):
    model.eval()
    img_tensor.requires_grad_(True)
    output = model(img_tensor)
    model.zero_grad()
    output.backward()

    grads = model.features.grad[0]
    fmap = model.features[0].detach()
    weights = grads.mean(dim=(1, 2))
    cam = (weights.view(-1, 1, 1) * fmap).sum(dim=0).detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() + 1e-6)
    cam = cv2.resize(cam, (128, 128))
    return cam, output.item()

# ----------------------------
# Heatmap Overlay
# ----------------------------
def overlay_heatmap(original_img, cam, threshold=0.4):
    mask = cam > threshold
    tumor_pixels = np.sum(mask)
    total_pixels = cam.size
    tumor_percent = (tumor_pixels / total_pixels) * 100

    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    return overlay, tumor_percent

# ----------------------------
# MRI Checker
# ----------------------------
def is_possible_mri(image_pil):
    grayscale = image_pil.convert("L")
    stat = ImageStat.Stat(grayscale)
    stddev = stat.stddev[0]
    entropy = grayscale.entropy()
    return 3.0 < entropy < 7.5 and 20 < stddev < 90

# ----------------------------
# Page config + Custom Styles
# ----------------------------
st.set_page_config(page_title="OncoVision", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background-color: #0f172a; }

      .oncovision-hero {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 18px;
      }
      .oncovision-title {
        color: white; font-size: 2rem; font-weight: 800; margin: 0;
      }
      .oncovision-sub {
        color: rgba(255,255,255,0.9); margin-top: 6px; font-size: 1rem;
      }

      /* Upload box */
      .upload-box {
        background: linear-gradient(90deg, #ff512f 0%, #dd2476 100%);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 18px;
      }
      .upload-title {
        font-size: 1.1rem; font-weight: 700; color: white; margin-bottom: 10px;
      }
      .upload-widget label { color: white !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
    <div class="oncovision-hero">
      <h1 class="oncovision-title">🧠 OncoVision</h1>
      <div class="oncovision-sub">Brain Tumor Detection with Tumor Localization</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Layout
# ----------------------------
left, right = st.columns([7, 5], gap="large")

with left:
    st.markdown('<div class="upload-box"><div class="upload-title">📂 Upload MRI Scan</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"], label_visibility="visible", key="upload1")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown("#### How it works")
    st.markdown(
        """
        - Upload a **brain MRI**.  
        - System checks if it looks like an MRI.  
        - If valid → tumor detection & heatmap shown.  
        - If invalid → warning only.  
        """
    )

# ----------------------------
# Load model
# ----------------------------
device = torch.device("cpu")
model = CNN_CAM().to(device)
model.load_state_dict(torch.load("model_weights_cam.pth", map_location=device))
model.eval()

# ----------------------------
# Inference
# ----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with left:
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if not is_possible_mri(image):
            st.warning("⚠️ This image does not look like a brain MRI. Please upload a proper scan.")
            st.stop()

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)

        cam, confidence = generate_cam(model, input_tensor)
        img_np = np.array(image.resize((128, 128)))
        overlay_img, tumor_area_percent = overlay_heatmap(img_np, cam)

        st.markdown("#### 🧪 Detection Result")
        if confidence > 0.5:
            st.error(f"**Tumor Detected** — Confidence: {confidence:.2%}")
            st.image(
                overlay_img,
                caption=f"Tumor Area Highlighted • Estimated Growth: {tumor_area_percent:.2f}%",
                use_container_width=True
            )
            st.progress(min(confidence, 1.0))
        else:
            st.success(f"**No Tumor Detected** — Confidence: {(1 - confidence):.2%}")
            st.progress(min(1 - confidence, 1.0))
