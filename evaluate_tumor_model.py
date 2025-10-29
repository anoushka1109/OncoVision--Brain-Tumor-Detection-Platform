import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob, cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Dataset loader & config
# ----------------------------
class MRI(Dataset):
    def __init__(self, path_yes, path_no, img_size=128):
        self.imgs, self.labels = [], []
        for f in glob.iglob(f"{path_yes}/*.jpg"):
            img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            self.imgs.append(img.transpose(2,0,1).astype(np.float32)/255.)
            self.labels.append(1)
        for f in glob.iglob(f"{path_no}/*.jpg"):
            img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            self.imgs.append(img.transpose(2,0,1).astype(np.float32)/255.)
            self.labels.append(0)
        self.imgs = np.stack(self.imgs)
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx): return {'image': self.imgs[idx], 'label': self.labels[idx]}

# ----------------------------
# CNN + Grad-CAM definition
# ----------------------------
class CNN_CAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*29*29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        self.features = x  # for Grad-CAM
        self.features.retain_grad()
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sig(self.fc3(x))

def grad_cam_heatmap(model, img_tensor):
    model.eval()
    out = model(img_tensor)
    prob = out.item()
    model.zero_grad()
    out.backward()
    grads = model.features.grad[0]
    fmap = model.features[0].detach()
    weights = grads.mean(dim=(1, 2))
    cam = (weights.view(-1, 1, 1) * fmap).sum(dim=0).detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img_tensor.shape[-1], img_tensor.shape[-1]))
    cam = cam - cam.min(); cam = cam / (cam.max() + 1e-6)
    return cam, prob

def mark_tumor_area(image, heatmap, thresh=0.4):
    mask = (heatmap > thresh).astype(np.uint8)
    tumor_area = mask.sum()
    total = mask.size
    perc = (tumor_area / total) * 100
    heat_rgb = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heat_rgb, 0.4, 0)
    return overlay, perc

# ----------------------------
# UI & evaluation
# ----------------------------
model = CNN_CAM().to('cpu')
model.load_state_dict(torch.load('model_weights_cam.pth', map_location='cpu'))

dataset = MRI("./data/brain_tumor_dataset/yes", "./data/brain_tumor_dataset/no")
loader = DataLoader(dataset, batch_size=1)

y_true, y_pred = [], []

for batch in loader:
    img = batch['image'].to(torch.float32)
    img = img.to('cpu')
    img.requires_grad_(True)

    label = batch['label'].item()
    cam, prob = grad_cam_heatmap(model, img)

    image_np = img[0].detach().cpu().numpy().transpose(1, 2, 0)
    image_np = (image_np * 255).astype(np.uint8)

    overlay, perc = mark_tumor_area(image_np, cam)

    pred = 1 if prob > 0.5 else 0
    y_true.append(label)
    y_pred.append(pred)

    title = f"Tumor" if pred == 1 else "Healthy"
    print(f"GT={label}, Pred={pred}, Conf={prob:.2%}, Tumor area ≈{perc:.2f}%")

    plt.figure(figsize=(4, 4))
    plt.imshow(overlay)
    plt.title(title)
    plt.axis("off")
    plt.show()

# Summary metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)
print(f"Precision: {precision_score(y_true, y_pred):.3f}, Recall: {recall_score(y_true, y_pred):.3f}, F1: {f1_score(y_true, y_pred):.3f}")