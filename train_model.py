import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import glob

# Dataset class
class MRI(Dataset):
    def __init__(self, paths_yes, paths_no, img_size=128):
        self.imgs, self.labels = [], []
        for f in glob.iglob(paths_yes + "/*.jpg"):
            img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            self.imgs.append(img.transpose(2, 0, 1).astype(np.float32)/255.)
            self.labels.append(1)
        for f in glob.iglob(paths_no + "/*.jpg"):
            img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            self.imgs.append(img.transpose(2, 0, 1).astype(np.float32)/255.)
            self.labels.append(0)

        self.imgs = np.stack(self.imgs)
        self.labels = np.array(self.labels, dtype=np.float32)

        # Train-val split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.imgs, self.labels, test_size=0.2, random_state=42
        )

        self.mode = 'train'

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        return len(self.X_train) if self.mode == 'train' else len(self.X_val)

    def __getitem__(self, idx):
        if self.mode == 'train':
            return {'image': self.X_train[idx], 'label': self.y_train[idx]}
        else:
            return {'image': self.X_val[idx], 'label': self.y_val[idx]}

# CNN Model with Grad-CAM support
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
        self.features = x  # For Grad-CAM
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sig(self.fc3(x))

# Load dataset
dataset = MRI("./data/brain_tumor_dataset/yes", "./data/brain_tumor_dataset/no")
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
dataset.set_mode('val')
val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Initialize model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_CAM().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    dataset.set_mode('train')
    train_losses = []

    for batch in train_loader:
        imgs = torch.tensor(batch['image']).to(device)
        labels = torch.tensor(batch['label']).to(device)

        optimizer.zero_grad()
        outputs = model(imgs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    dataset.set_mode('val')
    val_losses = []

    with torch.no_grad():
        for batch in val_loader:
            imgs = torch.tensor(batch['image']).to(device)
            labels = torch.tensor(batch['label']).to(device)
            outputs = model(imgs).squeeze()
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {np.mean(train_losses):.4f}, Val Loss: {np.mean(val_losses):.4f}")

# Save model
torch.save(model.state_dict(), "model_weights_cam.pth")
print("✅ Model saved as model_weights_cam.pth")
